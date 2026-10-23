const assert = require('node:assert/strict');
const Clock = require('../frontend/static/js/arena-timeline-clock.js');
const input = {steps:10, models:['svm','mlp','knn','tree','forest','bayes'], pointCount:180, trainCount:135,
    params:{svm:{max_iter:1000},mlp:{max_iter:500,solver:'adam',hidden_layer_sizes:'100,50'},
        knn:{n_neighbors:5},forest:{n_estimators:10},tree:{max_depth:5},bayes:{}}};
const estimate = values => Clock.estimateRecording({...input,...values});
const base = estimate({});
assert.equal(base.calibrated,0);
assert.ok(base.low>0 && base.high>base.low);
assert.ok(estimate({steps:1000}).high>base.high);
assert.ok(estimate({models:['knn','bayes']}).high<base.high);
assert.ok(estimate({pointCount:600,trainCount:450}).high>base.high);
assert.equal(estimate({steps:1000,models:['knn']}).tracks[0].count,5);
assert.equal(estimate({steps:1000,models:['mlp']}).tracks[0].count,500);
assert.ok(estimate({params:{...input.params,mlp:{...input.params.mlp,hidden_layer_sizes:'256,256,256'}}}).high>base.high);
const profile={setup_ms:2,recording_ms:26,complete:false,frames:[{duration_ms:1},{duration_ms:3}]};
const calibrated=estimate({steps:2,models:['svm'],profiles:{svm:profile}});
assert.equal(calibrated.calibrated,1);
assert.equal(calibrated.tracks[0].ms,26,'same frames reproduce measured recording cost');
assert.equal(estimate({steps:1,models:['svm'],profiles:{svm:profile}}).tracks[0].ms,13);
const extended=estimate({steps:4,models:['svm'],profiles:{svm:profile}}).tracks[0];
assert.ok(extended.ms>26,'independent budgets extrapolate compute, not just animation time');
assert.equal(estimate({steps:1000,models:['tree'],profiles:{tree:{...profile,complete:true}}}).tracks[0].count,2);
assert.equal(JSON.stringify(profile),JSON.stringify({setup_ms:2,recording_ms:26,complete:false,frames:[{duration_ms:1},{duration_ms:3}]}));
console.log('PASS recording estimates: step/model/data/network changes, caps, measured calibration, independent budgets');
