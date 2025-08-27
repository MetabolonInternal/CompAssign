# Public Datasets for Testing RT Prediction and Compound Assignment Models

## Overview

This document provides a comprehensive guide to publicly available metabolomics datasets that can be used to test and validate retention time (RT) prediction and compound assignment models. These datasets vary in their completeness, with most focusing on chemical standards rather than biological samples, presenting both opportunities and challenges for testing the CompAssign hierarchical Bayesian framework.

## Table of Contents
1. [Major Public Repositories](#major-public-repositories)
2. [Key Datasets for RT Prediction](#key-datasets-for-rt-prediction)
3. [Dataset Characteristics and Limitations](#dataset-characteristics-and-limitations)
4. [Integration Strategies for CompAssign](#integration-strategies-for-compassign)
5. [Chemical Descriptors and Embeddings](#chemical-descriptors-and-embeddings)
6. [Practical Implementation Guide](#practical-implementation-guide)

## Major Public Repositories

### 1. MetaboLights
- **URL**: https://www.ebi.ac.uk/metabolights/
- **Description**: European Bioinformatics Institute's metabolomics repository
- **Data Type**: Complete studies with raw data, processed results, and metadata
- **Strengths**: 
  - Contains biological context (species, tissue, conditions)
  - Follows ISA-Tab standards for metadata
  - FAIR compliant (Findable, Accessible, Interoperable, Reusable)
- **Limitations**: 
  - RT data quality varies between studies
  - Not all studies include chemical standards
  - Heterogeneous chromatographic conditions

### 2. GNPS (Global Natural Products Social Molecular Networking)
- **URL**: https://gnps.ucsd.edu/
- **Description**: Community-driven mass spectrometry knowledge base
- **Data Type**: MS/MS spectra, molecular networks, annotations
- **Strengths**:
  - Large-scale spectral libraries
  - Integration with MassIVE repository
  - Community annotations and molecular networking
- **Limitations**:
  - Limited RT information
  - Focus on MS/MS rather than chromatography

### 3. MassBank
- **URL**: https://massbank.eu/ (Europe), https://mona.fiehnlab.ucdavis.edu/ (North America)
- **Description**: Public repository of mass spectra with annotations
- **Data Type**: Reference MS and MS/MS spectra
- **Strengths**:
  - High-quality curated spectra
  - Standardized data format
  - Multiple ionization modes and collision energies
- **Limitations**:
  - RT data not consistently available
  - Primarily chemical standards, not biological samples

### 4. Metabolomics Workbench
- **URL**: https://www.metabolomicsworkbench.org/
- **Description**: NIH-funded national metabolomics repository
- **Data Type**: Complete metabolomics studies and reference data
- **Strengths**:
  - US complement to MetaboLights
  - RefMet compound database with standardized names
  - Integration with metabolic pathways
- **Limitations**:
  - Similar to MetaboLights in terms of RT data variability

## Key Datasets for RT Prediction

### 1. METLIN SMRT Dataset ⭐ (Highly Recommended)

**Publication**: Domingo-Almenara et al., Nature Communications, 2019
**DOI**: 10.1038/s41467-019-13680-7

**Access**: 
- Figshare: https://figshare.com/articles/dataset/The_METLIN_small_molecule_dataset_for_machine_learning-based_retention_time_prediction/8038913
- Free with citation requirement

**Dataset Characteristics**:
```
- Size: 80,038 small molecules
- Chromatography: Reversed-phase C18
- Column: ACQUITY UPLC BEH C18 (2.1 × 50 mm, 1.7 μm)
- Gradient: Water/ACN with 0.1% formic acid
- RT Range: 0-1200 seconds
- Temperature: 40°C
```

**Provided Data**:
- Retention times (seconds)
- PubChem CIDs
- SDF molecular structures
- Dragon 7 molecular descriptors (5,666 features)
- ECFP fingerprints (2,214 bits)
- SMILES strings

**Performance Benchmarks**:
- Best published MAE: 38.7 seconds (ChemProp GNN)
- Deep Neural Networks: 39.2 ± 1.2 seconds
- Random Forest baseline: ~60 seconds

**Advantages for CompAssign Testing**:
- Largest single-condition dataset available
- Standardized experimental conditions
- Pre-calculated molecular descriptors
- Well-established benchmark in the field

**Limitations**:
- No biological species information
- Single chromatographic system
- Pure chemical standards only
- No uncertainty quantification in ground truth

### 2. PredRet Database

**Publication**: Stanstrup et al., Analytical Chemistry, 2015
**DOI**: 10.1021/acs.analchem.5b02287

**Access**: 
- Website: http://www.predret.org
- API available for programmatic access

**Dataset Characteristics**:
```
- Compounds: 467+ unique molecules
- Systems: 24 different chromatographic conditions
- Coverage: 29-103 compounds per system
- Accuracy: Mean prediction error 2.6% (0.13 min)
```

**Source Studies from MetaboLights**:
- MTBLS4: Human urine metabolomics
- MTBLS17: Cardiac tissue metabolomics  
- MTBLS19: NMR and MS metabolomics comparison
- MTBLS20: Arabidopsis thaliana metabolomics
- MTBLS36: Human plasma lipidomics
- MTBLS38: Plant metabolite standards ⭐
- MTBLS39: Wine metabolomics
- MTBLS52: Tomato fruit metabolomics
- MTBLS87: HILIC chromatography standards

**Advantages**:
- Cross-system RT mapping capability
- Real experimental variation captured
- Both HILIC and reversed-phase data
- Community-driven data sharing

**Limitations**:
- Smaller size per chromatographic system
- Requires system-to-system projection models
- Incomplete compound coverage across systems

### 3. RepoRT Dataset

**Description**: Aggregated retention time database from multiple sources
**Sources**: SMRT, MassBank, MoNA, PredRet

**Characteristics**:
- Procedurally processed and harmonized
- Multiple chromatographic conditions
- Standardized compound identifiers
- Cross-referenced with multiple databases

### 4. MetaboLights Studies with RT Data

**High-Quality Studies for RT Modeling**:

**MTBLS38** (Plant Metabolite Standards):
- 200+ authentic standards
- Multiple concentration levels
- Complete MS and MS/MS data
- Used for MassBank reference spectra creation

**MTBLS20** (Arabidopsis thaliana):
- Plant metabolomics with species context
- 300+ identified metabolites
- Time-series data available
- Genetic variation included

**MTBLS87** (HILIC Standards):
- HILIC chromatography (complementary to C18)
- Polar metabolite coverage
- 150+ standards
- Alternative retention mechanism

## Dataset Characteristics and Limitations

### Chemical vs. Biological Context

| Dataset | Chemical Info | Biological Context | Species Data | RT Quality |
|---------|--------------|-------------------|--------------|------------|
| METLIN SMRT | ✅ Excellent | ❌ None | ❌ None | ✅ High |
| PredRet | ✅ Good | ❌ Limited | ❌ None | ✅ Good |
| MetaboLights | ✅ Variable | ✅ Rich | ✅ Available | ⚠️ Variable |
| GNPS | ✅ Good | ⚠️ Some | ⚠️ Some | ❌ Poor |
| MassBank | ✅ Excellent | ❌ None | ❌ None | ⚠️ Limited |

### Critical Gap: Species-Metabolite Associations

**The Challenge**:
- CompAssign uses hierarchical structure: Species → Clusters → Samples
- Public datasets mostly lack species information
- Chemical standards don't capture biological variation
- Matrix effects and species-specific metabolism not represented

**Why This Matters for CompAssign**:
```python
# CompAssign's hierarchical model expects:
hierarchical_structure = {
    "level_1": "species",      # Missing in most datasets
    "level_2": "individuals",   # Missing in most datasets  
    "level_3": "compounds",     # Available
    "level_4": "chemical_classes"  # Can be derived
}

# Public datasets typically provide:
available_structure = {
    "compounds": "identified",
    "retention_times": "measured",
    "chemical_structures": "SMILES/InChI",
    "descriptors": "calculated"
}
```

## Integration Strategies for CompAssign

### Strategy 1: Chemical Taxonomy as Pseudo-Hierarchy

**Approach**: Use chemical classification systems to create hierarchical structure

**Implementation**:
```python
# Using ClassyFire chemical taxonomy
chemical_hierarchy = {
    "Kingdom": ["Organic", "Inorganic"],
    "Superclass": ["Lipids", "Organic acids", "Alkaloids"],
    "Class": ["Fatty acids", "Amino acids", "Indoles"],
    "Subclass": ["Saturated FA", "Aromatic AA", "Simple indoles"]
}

# Map to CompAssign hierarchy
species_proxy = chemical_hierarchy["Kingdom"]
cluster_proxy = chemical_hierarchy["Superclass"]
compound_level = chemical_hierarchy["Subclass"]
```

**Tools**:
- ClassyFire API: http://classyfire.wishartlab.com/
- ChEBI ontology: https://www.ebi.ac.uk/chebi/
- HMDB classification: https://hmdb.ca/

### Strategy 2: Synthetic Species Assignment

**Approach**: Generate biologically plausible species-metabolite associations

**Method 1 - Pathway-Based**:
```python
metabolic_pathways = {
    "Plants": [
        "Flavonoid biosynthesis",
        "Alkaloid biosynthesis", 
        "Phenylpropanoid pathway",
        "Terpenoid biosynthesis"
    ],
    "Mammals": [
        "Bile acid metabolism",
        "Steroid hormone biosynthesis",
        "Amino acid metabolism",
        "Fatty acid beta-oxidation"
    ],
    "Bacteria": [
        "Peptidoglycan biosynthesis",
        "Quorum sensing",
        "Secondary metabolite production"
    ]
}
```

**Method 2 - Literature-Derived**:
- Use HMDB biospecimen data
- Parse MetaboLights studies for species-compound pairs
- Extract from pathway databases (KEGG, WikiPathways)

### Strategy 3: Multi-Dataset Fusion

**Approach**: Combine strengths of different datasets

```python
fusion_strategy = {
    "Step 1": "Use METLIN SMRT for comprehensive RT data",
    "Step 2": "Match compounds to MetaboLights studies",
    "Step 3": "Extract species context from MetaboLights",
    "Step 4": "Fill gaps with PredRet projections",
    "Step 5": "Add MS/MS from GNPS/MassBank"
}
```

**Implementation Pipeline**:
1. Load METLIN SMRT as base dataset
2. Query MetaboLights API for compound matches
3. Extract study metadata (species, tissue, condition)
4. Create hierarchical structure from metadata
5. Use PredRet for cross-system validation

### Strategy 4: Transfer Learning Approach

**Concept**: Train on chemical standards, adapt to biological samples

```python
transfer_learning_pipeline = {
    "Pretraining": {
        "dataset": "METLIN SMRT",
        "task": "RT prediction from structure",
        "model": "Base hierarchical model"
    },
    "Fine-tuning": {
        "dataset": "Selected MetaboLights studies",
        "task": "Species-specific RT adjustment",
        "model": "Adapted hierarchical model"
    },
    "Evaluation": {
        "dataset": "Held-out MetaboLights studies",
        "metrics": ["MAE", "Precision@95%", "Species-specific accuracy"]
    }
}
```

## Chemical Descriptors and Embeddings

### 1. Traditional Molecular Descriptors

**Constitutional Descriptors**:
- Molecular weight
- Atom counts (C, H, N, O, etc.)
- Bond counts (single, double, triple, aromatic)
- Ring counts
- Rotatable bond count

**Topological Descriptors**:
- Wiener index
- Zagreb indices
- Connectivity indices (Chi)
- Kappa shape indices
- Balaban index

**Geometric/3D Descriptors**:
- Molecular volume
- Surface area (SASA)
- Radius of gyration
- Moment of inertia
- Shape coefficients

**Electronic Descriptors**:
- LogP (hydrophobicity)
- LogD (distribution coefficient)
- pKa (acid dissociation)
- Dipole moment
- HOMO/LUMO energies
- Partial charges

**Software for Calculation**:
```python
# RDKit (open-source)
from rdkit import Chem
from rdkit.Chem import Descriptors
mol = Chem.MolFromSmiles(smiles)
mw = Descriptors.MolWt(mol)
logp = Descriptors.MolLogP(mol)

# Mordred (comprehensive)
from mordred import Calculator, descriptors
calc = Calculator(descriptors, ignore_3D=False)
results = calc(mol)

# PaDEL (Java-based)
# Dragon (commercial, used in METLIN)
# alvaDesc (commercial, extensive)
```

### 2. Molecular Fingerprints

**ECFP (Extended Connectivity Fingerprints)**:
```python
from rdkit.Chem import AllChem
# ECFP4 = radius 2, ECFP6 = radius 3
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
```

**MACCS Keys**:
```python
from rdkit.Chem import MACCSkeys
fp = MACCSkeys.GenMACCSKeys(mol)  # 166 structural keys
```

**Atom Pair Fingerprints**:
```python
from rdkit.Chem.AtomPairs import Pairs
fp = Pairs.GetAtomPairFingerprintAsBitVect(mol)
```

### 3. Modern Deep Learning Embeddings

**Graph Neural Networks (GNNs)**:

**ChemProp** (Message Passing Neural Network):
```python
# State-of-the-art for RT prediction
# 38.7s MAE on METLIN SMRT
from chemprop import train, predict
model = train(
    data_path='training_data.csv',
    dataset_type='regression',
    target_columns=['retention_time']
)
```

**AttentiveFP** (Graph Attention Network):
```python
# Attention mechanisms for interpretability
# Identifies important substructures for RT
```

**Transformer-Based Models**:

**ChemBERTa** (SMILES Transformer):
```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

# Pretrained on 77M SMILES from ZINC
# Fine-tune for RT prediction
```

**RT-Transformer** (2024):
- Specialized for RT prediction
- Combines structural and physicochemical features
- Attention weights show RT-relevant substructures

**MolFormer**:
```python
# Large-scale pretrained model
# 1.1B parameters, trained on 1.1B molecules
# Strong transfer learning capabilities
```

### 4. Hybrid Approaches for CompAssign

**Recommended Feature Set**:
```python
compassign_features = {
    "structural": {
        "source": "RDKit/Mordred",
        "features": ["MW", "LogP", "HBD", "HBA", "TPSA"],
        "dimension": ~200
    },
    "fingerprints": {
        "source": "ECFP4",
        "dimension": 2048,
        "use": "similarity calculation"
    },
    "embeddings": {
        "source": "ChemBERTa or ChemProp",
        "dimension": 768,
        "use": "rare compound handling"
    },
    "hierarchical": {
        "source": "ClassyFire API",
        "levels": ["kingdom", "superclass", "class", "subclass"],
        "use": "Bayesian hierarchy"
    }
}
```

**Handling Rare Compounds**:
```python
def get_compound_features(smiles, frequency_in_dataset):
    if frequency_in_dataset > 100:
        # Common compound: use traditional descriptors
        return calculate_descriptors(smiles)
    elif frequency_in_dataset > 10:
        # Uncommon: use fingerprints + descriptors
        return combine_features(
            calculate_descriptors(smiles),
            calculate_ecfp(smiles)
        )
    else:
        # Rare: use pretrained embeddings
        return get_pretrained_embedding(smiles)
```

## Practical Implementation Guide

### Step 1: Download and Prepare METLIN SMRT

```bash
# Download from Figshare
wget https://figshare.com/ndownloader/files/14554436 -O metlin_smrt.zip
unzip metlin_smrt.zip

# Files included:
# - SMRT_dataset.csv: Main RT data
# - descriptors/: Precalculated features
# - structures/: SDF files
```

```python
import pandas as pd
import numpy as np

# Load METLIN SMRT
df = pd.read_csv('SMRT_dataset.csv')
print(f"Total compounds: {len(df)}")
print(f"RT range: {df['RT'].min():.1f} - {df['RT'].max():.1f} seconds")
print(f"Missing values: {df.isnull().sum().sum()}")

# Convert to CompAssign format
compassign_data = {
    'compound_id': df['PubChem_CID'],
    'retention_time': df['RT'] / 60,  # Convert to minutes
    'smiles': df['SMILES'],
    # Add chemical hierarchy
    'chemical_class': classify_compounds(df['SMILES']),  # Use ClassyFire
    # Create synthetic species
    'species': assign_synthetic_species(df['SMILES'])
}
```

### Step 2: Access MetaboLights Studies

```python
import requests

# MetaboLights API
base_url = "https://www.ebi.ac.uk/metabolights/ws/studies"

# Get study list
response = requests.get(f"{base_url}/list")
studies = response.json()

# Filter for studies with RT data
rt_studies = []
for study_id in studies:
    study_data = requests.get(f"{base_url}/{study_id}").json()
    if 'retention_time' in str(study_data).lower():
        rt_studies.append(study_id)

# Download specific study (e.g., MTBLS38)
study_files = requests.get(f"{base_url}/MTBLS38/files").json()
# Download mzML files for raw data
# Download metadata sheets for compound identification
```

### Step 3: Use PredRet for Cross-System Mapping

```python
# PredRet doesn't have a public API, but you can:
# 1. Upload your data through web interface
# 2. Download predictions for your system
# 3. Or use the published data directly

# Load PredRet data (if downloaded)
predret_data = pd.read_csv('predret_export.csv')

# Create system mapping
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import PolynomialFeatures

def build_rt_projection(system1_rts, system2_rts):
    """Build projection model between chromatographic systems"""
    # GAM-like approach used by PredRet
    poly = PolynomialFeatures(degree=3)
    X = poly.fit_transform(system1_rts.reshape(-1, 1))
    
    # Isotonic regression for monotonic constraint
    iso_reg = IsotonicRegression(increasing=True)
    iso_reg.fit(system1_rts, system2_rts)
    
    return iso_reg
```

### Step 4: Create Test Dataset for CompAssign

```python
def create_compassign_test_data(source='metlin'):
    """
    Create test dataset with hierarchical structure
    """
    if source == 'metlin':
        # Load METLIN SMRT
        df = load_metlin_smrt()
        
        # Add chemical hierarchy
        df['kingdom'] = df['smiles'].apply(get_chemical_kingdom)
        df['superclass'] = df['smiles'].apply(get_chemical_superclass)
        df['class'] = df['smiles'].apply(get_chemical_class)
        
        # Create synthetic species based on chemical classes
        df['species'] = create_synthetic_species(df)
        
        # Add uncertainty (not in original data)
        df['rt_uncertainty'] = estimate_rt_uncertainty(df)
        
    elif source == 'metabolights':
        # Load MetaboLights studies with species
        df = load_metabolights_with_species(['MTBLS20', 'MTBLS38', 'MTBLS52'])
        
        # Already has biological hierarchy
        # Just need to standardize format
        
    # Create peak-compound pairs for assignment
    peak_compound_pairs = create_peak_compound_pairs(df)
    
    # Add mass accuracy features
    peak_compound_pairs['mass_error'] = calculate_mass_error(peak_compound_pairs)
    
    # Add RT difference features
    peak_compound_pairs['rt_diff'] = calculate_rt_diff(peak_compound_pairs)
    
    return df, peak_compound_pairs
```

### Step 5: Benchmark CompAssign Performance

```python
def benchmark_on_public_data():
    """
    Test CompAssign on public datasets
    """
    results = {}
    
    # Test on METLIN SMRT
    metlin_data, metlin_pairs = create_compassign_test_data('metlin')
    
    # Split data
    train_idx, test_idx = train_test_split(range(len(metlin_data)), test_size=0.2)
    
    # Train hierarchical RT model
    rt_model = train_hierarchical_rt_model(
        metlin_data.iloc[train_idx],
        use_species=False,  # No real species in METLIN
        use_chemical_hierarchy=True
    )
    
    # Predict on test set
    rt_predictions = rt_model.predict(metlin_data.iloc[test_idx])
    
    # Calculate metrics
    results['metlin'] = {
        'mae': mean_absolute_error(
            metlin_data.iloc[test_idx]['retention_time'],
            rt_predictions['mean']
        ),
        'rmse': np.sqrt(mean_squared_error(
            metlin_data.iloc[test_idx]['retention_time'],
            rt_predictions['mean']
        )),
        'r2': r2_score(
            metlin_data.iloc[test_idx]['retention_time'],
            rt_predictions['mean']
        )
    }
    
    # Test peak assignment
    assignment_model = train_peak_assignment_model(metlin_pairs[train_idx])
    assignment_preds = assignment_model.predict(metlin_pairs[test_idx])
    
    results['assignment'] = {
        'precision': precision_score(
            metlin_pairs[test_idx]['true_match'],
            assignment_preds > 0.9
        ),
        'recall': recall_score(
            metlin_pairs[test_idx]['true_match'],
            assignment_preds > 0.9
        )
    }
    
    return results
```

### Step 6: Cross-System Validation

```python
def cross_system_validation():
    """
    Test generalization across chromatographic systems
    """
    # Load PredRet data for multiple systems
    systems = load_predret_systems()
    
    results = []
    for train_system in systems:
        for test_system in systems:
            if train_system == test_system:
                continue
                
            # Train on one system
            model = train_compassign(systems[train_system])
            
            # Test on another system
            # Use PredRet projection for RT alignment
            projection = build_rt_projection(
                systems[train_system]['rt'],
                systems[test_system]['rt']
            )
            
            # Apply projection and test
            predictions = model.predict_with_projection(
                systems[test_system],
                projection
            )
            
            results.append({
                'train': train_system,
                'test': test_system,
                'mae': calculate_mae(predictions, systems[test_system]['rt'])
            })
    
    return pd.DataFrame(results)
```

## Expected Performance Metrics

### RT Prediction Performance

**METLIN SMRT Benchmarks**:
| Method | MAE (seconds) | RMSE (seconds) | R² |
|--------|--------------|----------------|-----|
| Random Forest | ~60 | ~85 | 0.91 |
| XGBoost | ~50 | ~72 | 0.93 |
| Deep Neural Network | 39.2 | 58.3 | 0.95 |
| ChemProp (GNN) | 38.7 | 57.1 | 0.95 |
| **CompAssign (expected)** | 45-55 | 65-75 | 0.92-0.94 |

**Note**: CompAssign may show higher MAE than pure deep learning models but should provide better uncertainty quantification and hierarchical borrowing for rare compounds.

### Peak Assignment Performance

**Expected on Public Data**:
| Dataset | Precision @ 0.9 threshold | Recall @ 0.9 threshold |
|---------|---------------------------|------------------------|
| METLIN (synthetic pairs) | 90-95% | 60-70% |
| MetaboLights (real) | 85-90% | 50-60% |
| Cross-system | 80-85% | 45-55% |

**Performance will be lower than synthetic data (99.5%) due to**:
- Real experimental noise
- Incomplete compound libraries
- Matrix effects not in training data
- Cross-system variations

## Recommendations for CompAssign Testing

### Priority 1: METLIN SMRT Validation
- Largest dataset for robust statistics
- Well-established benchmark
- Compare directly with published methods
- Focus on RT prediction component

### Priority 2: MetaboLights Biological Validation
- Select studies with species information (MTBLS20, MTBLS38, MTBLS52)
- Test hierarchical model advantages
- Validate uncertainty quantification
- Real-world applicability

### Priority 3: Cross-System Generalization
- Use PredRet for system transfer
- Test robustness to chromatographic variation
- Validate uncertainty increases appropriately
- Important for real-world deployment

### Priority 4: Rare Compound Performance
- Stratify test sets by compound frequency
- Compare performance degradation
- Validate hierarchical borrowing benefit
- Test with pretrained embeddings

## Future Directions

### 1. Active Learning with Public Data
```python
active_learning_pipeline = {
    "Step 1": "Train on high-confidence METLIN subset",
    "Step 2": "Predict on MetaboLights unknowns",
    "Step 3": "Select uncertain predictions for validation",
    "Step 4": "Iterate with community annotation"
}
```

### 2. Multi-Modal Integration
- Combine RT with MS/MS patterns (GNPS)
- Use CCS values from ion mobility (METLIN)
- Integrate NMR chemical shifts (HMDB)

### 3. Federated Learning Approach
- Train locally on proprietary Metabolon data
- Share model updates (not data) with public repositories
- Benefit from diverse data without sharing sensitive information

### 4. Benchmark Suite Development
Create standardized benchmark for compound assignment:
- Defined train/test splits
- Multiple difficulty levels
- Species-specific subsets
- Rare compound challenges

## References and Resources

### Key Papers
1. Domingo-Almenara et al. (2019). The METLIN small molecule dataset for machine learning-based retention time prediction. Nature Communications, 10(1), 5811.

2. Stanstrup et al. (2015). PredRet: Prediction of Retention Time by Direct Mapping between Multiple Chromatographic Systems. Analytical Chemistry, 87(18), 9421-9428.

3. Bouwmeester et al. (2019). Comprehensive and Empirical Evaluation of Machine Learning Algorithms for Small Molecule LC Retention Time Prediction. Analytical Chemistry, 91(5), 3694-3703.

4. Yang et al. (2024). RT-Transformer: Retention time prediction for metabolite annotation to assist in metabolite identification. Bioinformatics, 40(3), btae084.

### Useful APIs and Tools
- ClassyFire API: http://classyfire.wishartlab.com/
- MetaboLights API: https://www.ebi.ac.uk/metabolights/ws/api/spec
- ChemProp: https://github.com/chemprop/chemprop
- RDKit: https://www.rdkit.org/
- Mordred: https://github.com/mordred-descriptor/mordred

### Dataset Direct Links
- METLIN SMRT: https://figshare.com/articles/dataset/8038913
- PredRet: http://www.predret.org
- MetaboLights FTP: ftp://ftp.ebi.ac.uk/pub/databases/metabolights/
- GNPS Downloads: https://gnps-external.ucsd.edu/gnpslibrary
- MassBank GitHub: https://github.com/MassBank/MassBank-data

## Appendix: Data Format Examples

### METLIN SMRT Format
```csv
PubChem_CID,SMILES,RT,MW,LogP
2244,CC(=O)Oc1ccccc1C(=O)O,234.5,180.04,1.19
3672,CCc1ccccc1,156.8,106.08,3.15
```

### MetaboLights mzTab Format
```
MTD	mzTab-version	1.0.0
MTD	Study_title	Arabidopsis metabolomics
COM	Species: Arabidopsis thaliana
COM	Tissue: Leaf
SML	SML_ID	identifier	chemical_formula	smiles	retention_time
SML	1	HMDB0000148	C5H11NO2	CC(C)CC(N)C(=O)O	5.23
```

### CompAssign Expected Input Format
```python
{
    "compounds": {
        "compound_id": str,
        "smiles": str,
        "chemical_class": str,
        "retention_time": float
    },
    "hierarchy": {
        "species": str,          # May be synthetic
        "individual": str,       # May be missing
        "technical_replicate": str  # May be missing
    },
    "peak_compound_pairs": {
        "peak_id": str,
        "compound_id": str,
        "mass_error": float,
        "rt_diff": float,
        "is_match": bool  # Ground truth for testing
    }
}
```

## Conclusion

While public datasets provide valuable resources for testing RT prediction models, they present unique challenges for the CompAssign framework due to the lack of biological hierarchical structure. The recommended approach is to:

1. Start with METLIN SMRT for robust RT prediction benchmarking
2. Augment with MetaboLights for biological context where available
3. Use chemical taxonomy as a proxy for missing species information
4. Leverage modern embeddings for rare compound handling
5. Validate cross-system generalization with PredRet

The key insight is that CompAssign's hierarchical Bayesian approach is designed for biological complexity that most public datasets lack. Creative strategies like synthetic species assignment and chemical hierarchy mapping can bridge this gap, enabling meaningful validation on public data while maintaining the model's unique strengths.