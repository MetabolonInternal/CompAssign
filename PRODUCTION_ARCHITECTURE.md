# Production Architecture for Metabolomics Assignment

## Recommended Approach: Hybrid Pipeline

### Stage 1: Fast Candidate Filtering
```python
def get_candidates(peak, library, max_candidates=50):
    """
    Fast filtering based on mass and RT.
    Returns at most max_candidates compounds.
    """
    # Mass filter (very fast)
    mass_matches = library[
        abs(library.mass - peak.mz) < mass_tolerance
    ]
    
    # RT filter (fast) 
    rt_matches = mass_matches[
        abs(mass_matches.rt - peak.rt) < rt_window
    ]
    
    # If too many, rank by combined score
    if len(rt_matches) > max_candidates:
        rt_matches['score'] = (
            exp(-abs(mass_error) / mass_scale) * 
            exp(-abs(rt_error) / rt_scale)
        )
        rt_matches = rt_matches.nlargest(max_candidates, 'score')
    
    return rt_matches
```

### Stage 2: Probabilistic Assignment
```python
class ProductionAssignmentModel:
    def __init__(self, max_candidates=50):
        """
        Build model with fixed maximum dimension.
        50 candidates handles 99.9% of cases.
        """
        self.K_max = max_candidates + 1  # +1 for null
        self.model = self._build_model()
    
    def assign_batch(self, peaks, library):
        """
        Process peaks in batches for efficiency.
        """
        # Get candidates for each peak
        all_candidates = []
        for peak in peaks:
            candidates = get_candidates(peak, library, self.K_max - 1)
            all_candidates.append(candidates)
        
        # Pad to fixed dimension
        X, mask = self._prepare_tensors(all_candidates)
        
        # Run inference
        probs = self.model.predict(X, mask)
        
        return probs
```

### Stage 3: Adaptive Refinement (Optional)
```python
def adaptive_assignment(peak, library, model):
    """
    Start conservative, expand if needed.
    """
    # Try with top 20 candidates
    candidates = get_candidates(peak, library, max_candidates=20)
    probs = model.predict(candidates)
    
    # If confident, we're done
    if max(probs) > confidence_threshold:
        return candidates[argmax(probs)]
    
    # Otherwise, expand search
    candidates = get_candidates(peak, library, max_candidates=50)
    probs = model.predict(candidates)
    
    return candidates[argmax(probs)]
```

## Key Design Principles

1. **Separate Concerns**
   - Fast filtering (deterministic, mass/RT based)
   - Probabilistic scoring (learned, feature based)
   - Adaptive refinement (confidence based)

2. **Fixed Model Architecture**
   - Choose K_max based on 99th percentile of candidate counts
   - Typically 30-50 for metabolomics with reasonable tolerances
   - Use masking for peaks with fewer candidates

3. **Batch Processing**
   - Group similar peaks for GPU efficiency
   - Amortize model overhead across many peaks

4. **Fallback Strategy**
   - If a peak has >K_max candidates (rare):
     - Option A: Pre-filter to top K_max using simple scoring
     - Option B: Run multiple rounds with different candidate subsets
     - Option C: Flag for manual review

## Memory Considerations

For 10,000 peaks with K_max=50:
- Tensor size: 10,000 × 50 × n_features × 4 bytes
- With 10 features: ~20 MB (very manageable)
- Can process millions of peaks on a single GPU

## Implementation Priority

1. **MVP**: Fixed K_max=50, simple padding/masking
2. **V2**: Add adaptive refinement for low-confidence peaks  
3. **V3**: Dynamic batching by candidate count
4. **V4**: Multi-stage models with increasing complexity