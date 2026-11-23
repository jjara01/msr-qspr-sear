"""Preliminary surfactant screening for one contaminant, with reliability information.

For each candidate surfactant the table reports the predicted molar solubilization
ratio (MSR, mol contaminant per mol micellized surfactant) and the weight
solubilization ratio (WSR = MSR * MW_contaminant / MW_surfactant, g/g), with separate
rankings by MSR and by WSR. The inverse of WSR is reported as the mass of micellized
surfactant per mass of additional contaminant solubilized, under the assay
assumptions of the training measurements (above the CMC, equilibrium solubilization).
"""

import numpy as np
import pandas as pd

SCREENING_NOTICE = (
    'Predicted MSR is an equilibrium solubilization capacity estimated from literature '
    'measurements. It is not a prediction of aquifer remediation performance: sorption, '
    'phase behaviour, Krafft point, precipitation, toxicity, cost and site conditions are '
    'outside the model. Rankings support preliminary selection for experimental verification.')
SURFACTANT_COLUMNS = ('surfactant', 'smiles', 'cmc')
COLUMN_DEFINITIONS = {
    'rank_by_MSR': 'Rank by predicted MSR (1 = highest molar solubilization capacity).',
    'rank_by_WSR': 'Rank by predicted WSR (1 = highest mass solubilization capacity).',
    'predicted_log10_MSR': 'Predicted log10(MSR), MSR in mol contaminant per mol micellized surfactant.',
    'predicted_MSR_mol_per_mol': 'Predicted MSR, mol contaminant per mol micellized surfactant.',
    'predicted_WSR_g_per_g': 'Predicted WSR = MSR * MW_contaminant / MW_surfactant, g contaminant per g micellized surfactant.',
    'micellized_surfactant_g_per_g_additional_contaminant': (
        'Inverse WSR: mass of micellized surfactant per mass of additional contaminant solubilized, '
        'under the assay assumptions of the training measurements (above the CMC, equilibrium).'),
    'surfactant_molecular_weight_g_per_mol': 'Surfactant molecular weight used for WSR (supplied or from the structure).',
    'applicability_domain': 'Combined applicability status: inside, outside or incomplete.',
    'input_space_support': 'Chemical input-space support relative to the training structures.',
    'prediction_range': 'Whether the prediction lies within the training response range.',
    'local_reliability': 'Cross-fitted local-error status of the nearest training groups.',
    'abraham_descriptor_status': 'Provenance of the contaminant Abraham descriptors, when the model uses them.',
    'imputed_reference_properties': 'Reference inputs replaced by training medians.',
    'decision_support': 'Summary flag combining applicability and descriptor provenance.',
}


def molecular_weight(smiles):
    """Average molecular weight (g/mol) of the full represented structure."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    return float(Descriptors.MolWt(mol))


def _decision_support(ad_status, descriptor_status):
    if ad_status == 'outside':
        return 'outside_applicability_domain'
    if ad_status != 'inside' or descriptor_status not in {'not_required', 'curated_literature',
                                                          'curated_qspr_estimate', 'curated_estimate'}:
        return 'incomplete_reliability_information'
    return 'within_applicability_domain'


def rank_surfactants(predictor, *, contaminant_smiles, water_solubility, surfactants,
                     contaminant_type='Unknown', contaminant_name=None, contaminant_cas=None):
    """Rank surfactants for one contaminant with a fitted MSRPredictor.

    surfactants requires columns 'surfactant', 'smiles' and 'cmc' (mol/L); optional
    'surfactant_type' and 'molecular_weight' (g/mol, e.g. a nominal commercial value)
    columns are used when present. water_solubility is in g/L.
    """
    missing = set(SURFACTANT_COLUMNS) - set(surfactants)
    if missing or surfactants.empty or surfactants.surfactant.duplicated().any():
        raise ValueError(f'Screening requires unique surfactants with columns {SURFACTANT_COLUMNS}.')
    contaminant_weight = molecular_weight(contaminant_smiles)
    rows = []
    for record in surfactants.to_dict('records'):
        result = predictor.predict(
            cont_smiles=contaminant_smiles, surf_smiles=record['smiles'],
            cont_type=contaminant_type, surf_type=record.get('surfactant_type', 'Unknown'),
            cmc=None if pd.isna(record['cmc']) else record['cmc'], water_solubility=water_solubility,
            contaminant_name=contaminant_name, contaminant_cas=contaminant_cas)
        weight = record.get('molecular_weight')
        weight = molecular_weight(record['smiles']) if weight is None or pd.isna(weight) else float(weight)
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError('Surfactant molecular weight must be finite and positive.')
        wsr = result.msr * contaminant_weight / weight
        rows.append({
            'surfactant': record['surfactant'],
            'predicted_log10_MSR': result.log10_msr,
            'predicted_MSR_mol_per_mol': result.msr,
            'predicted_WSR_g_per_g': wsr,
            'micellized_surfactant_g_per_g_additional_contaminant': 1 / wsr,
            'surfactant_molecular_weight_g_per_mol': weight,
            'applicability_domain': result.ad.status,
            'input_space_support': result.ad.structural.availability_status,
            'prediction_range': result.ad.y_range.availability_status,
            'local_reliability': result.ad.local_error.availability_status,
            'abraham_descriptor_status': result.abraham_descriptor_status,
            'imputed_reference_properties': ', '.join(result.imputed_reference_properties),
            'decision_support': _decision_support(result.ad.status, result.abraham_descriptor_status),
        })
    table = pd.DataFrame(rows)
    table.insert(1, 'rank_by_WSR', table.predicted_WSR_g_per_g.rank(ascending=False, method='min').astype(int))
    table.insert(1, 'rank_by_MSR', table.predicted_log10_MSR.rank(ascending=False, method='min').astype(int))
    return table.sort_values(['rank_by_MSR', 'surfactant'], kind='stable').reset_index(drop=True)
