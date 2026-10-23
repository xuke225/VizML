import copy
import unittest
from unittest.mock import patch

from flask import Flask

from backend.arena.api import blueprint
from backend.arena.engine import generate, fit_experiment, selected_model_keys
from backend.arena.observation import fit_observation
from backend.arena.timeline import record_timeline


class ArenaSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload=dict(points=generate({"count":48})["points"],seed=42,step=2,steps=2,
                         params={"mlp":{"hidden_layer_sizes":"8,4"}})

    def test_default_all_and_validation(self):
        self.assertEqual(len(selected_model_keys({})),6)
        self.assertEqual(selected_model_keys({"selected_models":["mlp","svm"]}),('svm','mlp'))
        for selection in (None,[],['svm'],['svm','svm'],['svm','alien'],['svm',[]],True,'svm,knn',list(range(7))):
            with self.subTest(selection=selection),self.assertRaises(ValueError):
                selected_model_keys({"selected_models":selection})

    def test_two_through_six_participants(self):
        from backend.arena.engine import MODEL_KEYS
        for count in range(2,7):
            result=fit_experiment(dict(self.payload,selected_models=list(MODEL_KEYS[:count])))
            self.assertEqual(len(result['models']),count)
            self.assertEqual(set(result['ranks']),set(MODEL_KEYS[:count]))
            self.assertEqual(set(result['model_keys']),set(MODEL_KEYS[:count]))

    def test_inactive_models_are_not_constructed_or_validated(self):
        import backend.arena.engine as engine
        actual=engine.create_classifier
        constructed=[]
        def create(key,*args):
            constructed.append(key)
            return actual(key,*args)
        payload=copy.deepcopy(self.payload)
        payload.update(selected_models=['svm','knn'],params={'mlp':{'not_a_setting':999},'forest':{'n_estimators':-1}})
        for action in (fit_experiment,fit_observation,record_timeline):
            constructed.clear()
            with patch.object(engine,'create_classifier',side_effect=create):
                result=action(payload)
            self.assertEqual(constructed,['svm','knn'])
            self.assertEqual(result['model_keys'],['svm','knn'])

    def test_selection_does_not_change_data_split_or_selected_predictions(self):
        all_models=fit_experiment(self.payload)
        selected=fit_experiment(dict(self.payload,selected_models=['knn','svm']))
        self.assertEqual(selected['split'],all_models['split'])
        self.assertEqual(selected['holdout_fingerprint'],all_models['holdout_fingerprint'])
        for key in selected['models']:
            self.assertEqual(selected['models'][key]['grid'],all_models['models'][key]['grid'])
            self.assertEqual(selected['models'][key]['test'],all_models['models'][key]['test'])

    def test_observation_and_replay_only_include_selected(self):
        for selected in (['svm','knn'],['forest','mlp'],['tree','bayes','mlp']):
            p=dict(self.payload,selected_models=selected)
            stage=fit_observation(p)
            timeline=record_timeline(p)
            self.assertEqual(set(stage['models']),set(selected))
            self.assertEqual(set(timeline['tracks']),set(selected))
            self.assertEqual(stage['split'],timeline['split'])
            for key in selected:
                frame=timeline['tracks'][key]['frames'][-1]
                self.assertEqual(frame['model']['predictions'],stage['models'][key]['predictions'])

    def test_endpoints_require_two_and_return_only_requested_models(self):
        app=Flask(__name__);app.register_blueprint(blueprint)
        with app.test_client() as client:
            for route in ('run','observe','timeline'):
                result=client.post('/api/arena/'+route,json=dict(self.payload,selected_models=['forest','mlp']))
                self.assertEqual(result.status_code,200)
                self.assertEqual(set(result.json.get('models',result.json.get('tracks'))),{'forest','mlp'})
                self.assertEqual(client.post('/api/arena/'+route,json=dict(self.payload,selected_models=['svm'])).status_code,400)


if __name__=='__main__':
    unittest.main()
