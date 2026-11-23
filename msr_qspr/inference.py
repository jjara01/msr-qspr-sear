"""Fitted-bundle inference with serialized training-only reliability references."""

from __future__ import annotations

import warnings
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from msr_qspr.config import DATA_DIR
ROOT = DATA_DIR.parent


def _resolve_artifacts_dir(root: Path) -> Path:
    """Resolve artifact directory for app loading."""
    env_dir = os.getenv("MSR_ARTIFACTS_DIR")
    if env_dir:
        return Path(env_dir)

    return root / "artifacts"


ARTIFACTS = _resolve_artifacts_dir(ROOT)


def report_table(artifacts: Path, filename: str):
    """Verified stage table for the selected final configuration, or None if unavailable."""
    from msr_qspr.reporting import selected_report_table
    return selected_report_table(artifacts, filename)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ADCriterion:
    name: str
    description: str
    value: float
    threshold: float
    pass_: bool
    value_label: str = ""
    threshold_label: str = ""
    calibration_status: str = "unavailable"
    availability_status: str = "unavailable"


@dataclass
class ADResult:
    structural: ADCriterion
    y_range: ADCriterion
    local_error: ADCriterion
    overall: bool
    neighbor_indices: np.ndarray
    neighbor_dists: np.ndarray
    calibration_status: str = "incomplete"
    status: str = "incomplete"


@dataclass
class PredictionResult:
    log10_msr: float
    msr: float
    ad: ADResult
    shap_values: np.ndarray
    shap_base: float
    feature_names: list
    feature_values: np.ndarray
    imputed_reference_properties: tuple[str, ...] = ()
    shap_status: str = 'available'
    shap_reason: str = ''
    abraham_descriptor_status: str = 'not_required'
    imposed_offset: float = float('nan')


# ---------------------------------------------------------------------------
# Main predictor class
# ---------------------------------------------------------------------------

class MSRPredictor:
    """
    Loads all necessary artifacts and computes:
      - log10(MSR) prediction via the trained reporting pipeline
      - Applicability Domain (3 criteria, exact replication of diagnostics.py)
      - SHAP explanation for the prediction
    """

    def __init__(self, root: Path = ROOT, *, abraham_directory=None, abraham_provider=None):
        """abraham_provider optionally estimates descriptors for uncurated contaminants."""
        self.root = root
        self.artifacts = _resolve_artifacts_dir(root)
        self.abraham_directory = abraham_directory
        self.abraham_provider = abraham_provider
        self._abraham_panel = None
        self._load()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Inference only: all preprocessing and reliability references must be serialized."""
        from msr_qspr.final import load_inference_bundle
        from msr_qspr.artifacts import bundle_directory
        self.pipe, self.bundle = load_inference_bundle(bundle_directory(self.artifacts))
        if self.bundle.get('analysis_support', {}).get('reliability') != 'available':
            raise ValueError('This predictor supports direct predict(X). Application reliability and interpretation '
                             'require separate validation before use.')
        self.reference = self.pipe.reliability_
        self.preproc = Pipeline(self.pipe.steps[:-1])
        self.model = self.pipe.steps[-1][1]
        self.feature_names = self.reference.features
        self.required_reference_properties = tuple(self.bundle['inference']['required_reference_properties'])
        fitted_properties = tuple(name for name in ('CMC', 'WaterSolubility')
                                  if name in self.pipe.named_steps['schema'].columns_)
        if self.required_reference_properties != fitted_properties:
            raise ValueError('Bundle reference-property declaration differs from the fitted predictor schema.')
        self._y_cv_log, self._meta_cv = self.reference.y_log, self.reference.metadata
        values = self.pipe.interpretation_background_
        self._X_cv_pre = values
        self._feature_stats = pd.DataFrame({'feature': self.feature_names,
            'train_mean': values.mean(axis=0), 'train_std': values.std(axis=0),
            'train_min': values.min(axis=0), 'train_max': values.max(axis=0)})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_smiles(self, smiles: str, *, surfactant: bool = False):
        """Validate contaminants permissively and surfactants by the grouping contract."""
        if surfactant:
            from msr_qspr.validation import _active_structure
            try:
                return True, _active_structure(str(smiles).strip())
            except ValueError as error:
                return False, (f"{error} Expected one carbon-containing fragment, supported "
                               "counterions, and charge balance when counterions are present.")
        from msr_qspr.chemistry import canonical_main_organic_fragment
        canon = canonical_main_organic_fragment(str(smiles).strip())
        if canon is None:
            return False, "Could not parse SMILES"
        return True, canon

    def predict(
        self,
        cont_smiles: str,
        surf_smiles: str,
        cont_type: str = "Unknown",
        surf_type: str = "Unknown",
        cmc: Optional[float] = None,
        water_solubility: Optional[float] = None,
        contaminant_name: Optional[str] = None,
        contaminant_cas: Optional[str] = None,
    ) -> PredictionResult:
        """
        Full prediction pipeline:
          1. Compute Mordred descriptors from SMILES
          2. Predict log10(MSR) via the trained pipeline
          3. Check applicability domain (3 criteria)
          4. Compute SHAP values for interpretation

        cmc is an independent reference in mol/L; water_solubility is in g/L.
        Missing reference properties use only the stored training imputation.
        Abraham descriptors come from the curated table, then the optional provider;
        unresolved contaminants are imputed from training and flagged. The curated
        name or CAS number is needed for stereoisomers that a structure cannot identify.
        """
        input_row = self._build_input_row(
            cont_smiles, surf_smiles, cont_type, surf_type, cmc, water_solubility,
            contaminant_name, contaminant_cas,
        )
        descriptor_status = input_row.attrs.get('abraham_descriptor_status', 'not_required')
        imputed = tuple(name for name in ('CMC', 'WaterSolubility')
                        if name in input_row and pd.isna(input_row[name].iloc[0]))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log10_msr = float(self.pipe.predict(input_row)[0])
            X_pre = np.asarray(self.preproc.transform(input_row), dtype=float)

        from msr_qspr.reliability import fitted_shap
        from msr_qspr.validation import _active_structure
        from msr_qspr.chemistry import canonicalize
        metadata = pd.DataFrame({'row_id': ['application_query'],
            'active_structure_group': [_active_structure(surf_smiles)],
            'contaminant_group': [canonicalize(cont_smiles, True)]})
        assessment = self.reference.assess(self.pipe, input_row, metadata,
                                           unavailable_properties=tuple(p for p in imputed
                                               if p in self.reference.features)).iloc[0]
        ad = self._compute_ad(assessment)
        explanation = fitted_shap(self.pipe, input_row, None)
        if explanation['status'] == 'available':
            sv = explanation['values'][0]
            base = float(np.asarray(explanation['base_values']).reshape(-1)[0])
        else:
            sv, base = np.full(len(self.feature_names), np.nan), float('nan')

        return PredictionResult(
            log10_msr=log10_msr,
            msr=10.0 ** log10_msr,
            ad=ad,
            shap_values=sv,
            shap_base=base,
            feature_names=self.feature_names,
            feature_values=X_pre[0],
            imputed_reference_properties=imputed,
            shap_status=explanation['status'],
            shap_reason=explanation.get('reason', ''),
            abraham_descriptor_status=descriptor_status,
            imposed_offset=(float(explanation['components']['imposed_offset'][0])
                            if 'components' in explanation else float('nan')),
        )

    def get_neighbors(self, ad: ADResult) -> pd.DataFrame:
        """Return DataFrame with k nearest training compounds."""
        rows = []
        for rank, (i, dist) in enumerate(zip(ad.neighbor_indices, ad.neighbor_dists)):
            row: dict = {
                "Rank": rank + 1,
                "Distance": round(float(dist), 3),
                "log₁₀(MSR)": round(float(self._y_cv_log[i]), 3),
                "MSR": round(float(10.0 ** self._y_cv_log[i]), 5),
            }
            if i < len(self._meta_cv):
                meta_row = self._meta_cv.iloc[i]
                for col in ["Contaminant", "Surfactant", "SMILES_contaminant", "SMILES_surfactant"]:
                    if col in meta_row.index:
                        row[col] = meta_row[col]
            rows.append(row)
        return pd.DataFrame(rows)

    def get_feature_stats(self) -> pd.DataFrame:
        """Return cached training feature statistics for model descriptors."""
        return self._feature_stats.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_input_row(
        self,
        cont_smiles: str,
        surf_smiles: str,
        cont_type: str,
        surf_type: str,
        cmc: Optional[float],
        water_solubility: Optional[float],
        contaminant_name: Optional[str] = None,
        contaminant_cas: Optional[str] = None,
    ) -> pd.DataFrame:
        """Validate both molecular roles before assembling any predictor inputs.

        The Abraham descriptor resolution status is recorded in the returned frame's attrs.
        """
        for role, smiles in (("contaminant", cont_smiles), ("surfactant", surf_smiles)):
            valid, message = self.validate_smiles(smiles, surfactant=role == "surfactant")
            if not valid:
                raise ValueError(f"Invalid {role} input: {message}")
        from msr_qspr.chemistry import add_descriptors
        from msr_qspr.preprocessing import build_predictor_matrix
        schema = self.pipe.named_steps["schema"]
        for name, value in (('CMC', cmc), ('WaterSolubility', water_solubility)):
            if name in schema.columns_ and value is not None:
                try:
                    valid = np.isfinite(float(value)) and float(value) > 0
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    raise ValueError(f'{name} must be a finite positive reference value in ' +
                                     ('mol/L.' if name == 'CMC' else 'g/L.'))
        base: dict = {
            "ContaminantType": cont_type,
            "SurfactantType": surf_type,
            "CMC": cmc if 'CMC' in schema.columns_ and cmc is not None else np.nan,
            "WaterSolubility": water_solubility if 'WaterSolubility' in schema.columns_ and water_solubility is not None else np.nan,
            "SMILES_contaminant": cont_smiles.strip(),
            "SMILES_surfactant": surf_smiles.strip(),
        }
        if schema.counterion_policy_ is None:
            row = pd.DataFrame([base])
        else:
            row = add_descriptors(pd.DataFrame([base]), counterion_policy=schema.counterion_policy_)
        status = 'not_required'
        from msr_qspr.abraham import ABRAHAM_DESCRIPTORS, STATUS_COLUMN, attach_descriptors, load_panel
        if set(ABRAHAM_DESCRIPTORS) & set(schema.columns_):
            if self._abraham_panel is None:
                self._abraham_panel = load_panel(self.abraham_directory)
            row = row.assign(Contaminant=contaminant_name, contaminant_cas=contaminant_cas)
            row = attach_descriptors(row, panel=self._abraham_panel, provider=self.abraham_provider)
            status = row[STATUS_COLUMN].iloc[0]
            row = row.drop(columns=['Contaminant', 'contaminant_cas', STATUS_COLUMN])
        X = build_predictor_matrix(row)
        if not set(schema.columns_) <= set(X):
            raise ValueError("Inference cannot supply the fitted predictor schema.")
        result = X.loc[:, list(schema.columns_)].copy()
        result.attrs['abraham_descriptor_status'] = status
        return result

    def _compute_ad(self, row) -> ADResult:
        def criterion(name, description, state, value, threshold, calibration):
            available = state in {'inside', 'outside'}
            return ADCriterion(name=name, description=description,
                value=float(value) if value is not None else float('nan'),
                threshold=float(threshold) if threshold is not None else float('nan'),
                pass_=state == 'inside', calibration_status=calibration, availability_status=state,
                value_label=(f'{value:.3f}' if available and value is not None else
                             'Unavailable'),
                threshold_label=f'≤ {threshold:.3f}' if threshold is not None else 'Unavailable')
        lookup = {obs: i for i, obs in enumerate(self.reference.ids)}
        structural = criterion('Input-space Support', row.chemical_reason, row.chemical_support,
            row.distance, row.distance_threshold, 'training_derived')
        response = criterion('Prediction Range', row.response_reason, row.response_range,
            row.predicted_log10_MSR, None, 'training_derived')
        response.threshold_label = f'[{row.response_min:.2f}, {row.response_max:.2f}]'
        local = criterion('Local Reliability', row.local_error_reason, row.local_error,
            row.local_error_value, row.local_error_threshold, row.local_calibration_status)
        return ADResult(structural=structural, y_range=response, local_error=local,
            overall=row.ad_status == 'inside',
            neighbor_indices=np.array([lookup[obs] for obs in row.neighbor_ids], dtype=int),
            neighbor_dists=np.asarray(row.neighbor_distances),
            calibration_status=row.local_calibration_status,
            status=row.ad_status)
