"""Pruebas de integridad, métricas y restricciones del caso real."""
from pathlib import Path
import hashlib
import json
import unittest
import joblib
import numpy as np
from routing import solve_pilot

ROOT=Path(__file__).resolve().parent


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d={i:joblib.load(ROOT/'artifacts'/f'dim{i}.joblib') for i in range(1,5)}

    def test_data_provenance(self):
        m=json.loads((ROOT/'artifacts/manifest.json').read_text(encoding='utf-8'))
        for name,digest in m['files'].items():
            self.assertEqual(hashlib.sha256((ROOT/'datos'/name).read_bytes()).hexdigest(),digest)

    def test_reference_metrics_from_notebooks(self):
        self.assertAlmostEqual(self.d[1]['metrics']['MAE'],1.7130,delta=.08)
        self.assertAlmostEqual(self.d[1]['metrics']['R²'],.9733,delta=.002)
        self.assertEqual(self.d[3]['alpha'],100.)
        ridge=self.d[3]['comparison'].query("Modelo == 'Ridge'").iloc[0]
        self.assertAlmostEqual(ridge.MAE,11.6005,delta=.005)
        sarima=self.d[4]['comparison'].query("Modelo == 'SARIMA con constante'").iloc[0]
        # Independent execution of the supplied SARIMA cell with pinned versions.
        # The report's 41.70 was not reproduced; see OBSERVACIONES_INFORME.md.
        self.assertAlmostEqual(sarima.MAE,44.6845,delta=.02)
        self.assertEqual(int(self.d[4]['monthly'].sum()),85333)
        self.assertEqual(self.d[4]['capacity'],2392)

    def test_pilot_reference_and_coverage(self):
        p=self.d[2]['pilot'];r=solve_pilot(p)
        self.assertEqual(int(p.weight_lbs.sum()),105569)
        expected={3:7235.556,5:8365.939}
        for n in [3,5]:
            self.assertEqual(sorted(i for x in r['paths'][n,'Optimizada'] for i in x),list(range(6)))
            self.assertEqual(len(r['paths'][n,'Optimizada']),n)
            row=r['comparison'].query('Camiones == @n').iloc[0]
            self.assertLessEqual(row['Optimizada (mi)'],row['Referencia (mi)']+1e-6)
            self.assertAlmostEqual(row['Optimizada (mi)'],expected[n],delta=.01)
        self.assertTrue((r['details']['Carga (lb)']<=45000).all())

    def test_infeasible_capacity_and_time_sensitivity(self):
        p=self.d[2]['pilot']
        bad=solve_pilot(p,capacity=1000)
        self.assertTrue(bad['details'].empty)
        a,b=solve_pilot(p,speed=50,service=0),solve_pilot(p,speed=25,service=0)
        np.testing.assert_allclose(b['details']['Tiempo estimado (h)'],a['details']['Tiempo estimado (h)']*2)

    def test_horizons_and_train_test_sizes(self):
        self.assertEqual((self.d[3]['n_train'],self.d[3]['n_test']),(100,52))
        self.assertEqual(len(self.d[4]['future']),12)
        self.assertTrue((self.d[4]['future'].index.year==2025).all())
        self.assertTrue(np.isfinite(self.d[4]['future'].to_numpy()).all())
        self.assertTrue((self.d[4]['future']['Límite superior 95%']>=self.d[4]['future']['Límite inferior 95%']).all())


if __name__=='__main__':
    unittest.main(verbosity=2)
