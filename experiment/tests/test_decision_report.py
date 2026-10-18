"""Protect information boundaries and fallback interpretation in manager exports."""
import importlib.util
from pathlib import Path
import unittest
import pandas as pd

PATH = Path(__file__).resolve().parents[1]/'scripts/operations/export_decision_report.py'
SPEC = importlib.util.spec_from_file_location('decision_report', PATH)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class DecisionReportTests(unittest.TestCase):
    def test_fallback_uses_reference_prediction_not_rejected_candidate(self):
        r={'policy':'GA','candidate_policy':'PPO','reference_policy':'GA',
           'predicted_cost':10.,'predicted_service':.5,
           'reference_predicted_cost':20.,'reference_predicted_service':.8}
        self.assertEqual(report.final_prediction(r),(20.,.8,'reference_prediction'))
        r['policy']='PPO'
        self.assertEqual(report.final_prediction(r),(10.,.5,'candidate_prediction'))

    def test_missing_predictions_remain_unavailable(self):
        self.assertEqual(report.final_prediction({'policy':'GA'}),
                         (None,None,'not_persisted_for_this_strategy'))

    def test_held_out_diagnostic_labels_do_not_change_decision_export(self):
        panel=pd.DataFrame([[2.,3.,4.,5.]],index=pd.MultiIndex.from_tuples(
            [('BA','seller','item')],names=['warehouse','store_id','item_id']),
            columns=['c1','c2','c3','c4'])
        predictions=pd.DataFrame([{'origin':2,'warehouse':'BA','store_id':'seller',
            'item_id':'item','strategy':'Global','policy':'GA','validation_label':'PPO',
            'label_unseen_in_training':False}])
        profiles={(2,'BA','seller','item'):'regular'}
        first=report.make_decisions(predictions,{2:panel},profiles,'example')
        predictions['validation_label']='DQN'
        predictions['label_unseen_in_training']=True
        second=report.make_decisions(predictions,{2:panel},profiles,'example')
        self.assertEqual(first,second)
        # Future realized sales must not enter either the recorded history or explanation.
        panel.iloc[:,2:]=99999.
        third=report.make_decisions(predictions,{2:panel},profiles,'example')
        self.assertEqual(first,third)
        self.assertNotIn('validation_label',first[0])
        self.assertIsNone(first[0]['policy_parameters'])
        self.assertFalse(first[0]['ready_for_purchase_order'])

    def test_embedded_json_cannot_close_script_element(self):
        text=report.build_html([{'strategy':'</script><script>alert(1)</script>'}],[])
        self.assertIn('\\u003c/script>',text)
        self.assertNotIn('alert(1)</script>',text)


if __name__ == '__main__':
    unittest.main()
