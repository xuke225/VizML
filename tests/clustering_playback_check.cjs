// Run with: node tests/clustering_playback_check.cjs
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(require('node:path').join(__dirname, '../frontend/templates/clustering.html'), 'utf8');
const script = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).join('\n');
const centers = [];
const context = {canvas:{width:600,height:320},ellipse(){},setLineDash(){},beginPath(){},moveTo(){},lineTo(){},stroke(){},fill(){},fillRect(){},clearRect(){},save(){},restore(){},fillText(){},arc(x,y,r){if(r===15)centers.push([x,y]);}};
const elements = new Map();
const ready = [], raf = new Map(); let nextRaf = 1;
const sandbox = {window:{},console,clearTimeout,setTimeout,cancelAnimationFrame(id){raf.delete(id);},requestAnimationFrame(fn){const id=nextRaf++;raf.set(id,fn);return id;},performance:{now:()=>0}, document:{addEventListener(name,fn){if(name === "DOMContentLoaded") ready.push(fn);},getElementById(id){if(!elements.has(id))elements.set(id,{value:700,width:600,height:320,addEventListener(){},getContext:()=>context});return elements.get(id);}}};
vm.createContext(sandbox);vm.runInContext(script,sandbox);
vm.runInContext(`
window.dataRange={xMin:10,xMax:20,yMin:30,yMax:40};
resetStepControls();
if (!window.dataRange) throw new Error('Dataset reset discarded center projection range');
data=[{x:.5,y:.5}];
iterationFrames=[
 {title:'init',description:'',iteration:0,labels:[0],centers:[[12,32]],metric:10},
 {title:'update',description:'',iteration:1,labels:[0],centers:[[18,38]],metric:5}
];
iterationIndex=1;
renderIteration(0,.5);
`,sandbox);
assert.ok(centers.length>0);
assert.deepEqual(centers.at(-1),[300,160], 'Halfway animation must project the interpolated centroid inside the canvas');
console.log('PASS: coordinate range preserved; interpolated centroid rendered at canvas midpoint');

vm.runInContext(`
for (const frame of [
 {algorithm:'dbscan',labels:[-2],centers:[],focus:0,eps:1,neighbors:[0]},
 {algorithm:'hierarchical',labels:[7],centers:[],highlight:[0]},
 {algorithm:'gmm',labels:[0],centers:[[15,35]],confidence:[.6],covariances:[[[1,0],[0,1]]]}
]) {
 iterationFrames=[{title:'test',description:'test',iteration:1,metric:1,...frame}];
 iterationIndex=0; renderIteration();
}
`,sandbox);
console.log('PASS: DBSCAN, hierarchical and GMM frames render without K-Means assumptions');

ready[0]();
vm.runInContext(`
data=[{x:.5,y:.5}];
iterationFrames=[
 {title:'start',description:'',iteration:0,labels:[0],centers:[[12,32]],metric:10},
 {title:'end',description:'',iteration:1,labels:[0],centers:[[18,38]],metric:5}
];
iterationIndex=0; renderIteration();
`,sandbox);
const seek=elements.get('iterationSeek');
seek.value='1'; seek.oninput({target:seek});
assert.equal(vm.runInContext('iterationIndex',sandbox),1,'Dragging must read the requested step before pause redraws the slider');
assert.equal(elements.get('iterationCount').textContent,'2 / 2');
elements.get('iterationPrev').onclick();
assert.equal(elements.get('iterationCount').textContent,'1 / 2','Single step must update controls immediately');
for (const [id, fn] of [...raf]) {raf.delete(id);fn(1000);}
assert.equal(elements.get('iterationCount').textContent,'1 / 2');
vm.runInContext('data=[];clearIterationHistory()',sandbox);
assert.equal(elements.get('iterationCount').textContent,'0 / 0');
assert.equal(elements.get('iterationSeek').disabled,true);
assert.equal(raf.size,0);
console.log('PASS: drag progress, immediate backward step, pending-animation cancellation, dataset reset');

(async()=>{
 vm.runInContext("currentSessionKey='old';apiClient={trainClustering:()=>new Promise(resolve=>window.resolveTraining=resolve)};",sandbox);
 vm.runInContext('getAlgorithmParams=()=>({})',sandbox);
 const pending=vm.runInContext("getFinalClusteringResults('dbscan')",sandbox);
 vm.runInContext("clearIterationHistory();currentSessionKey='new';window.resolveTraining({status:'success',results:{labels:[0]}})",sandbox);
 assert.equal(await pending,false,'An old dataset response must not replace the current view');
 console.log('PASS: stale clustering response ignored after data changes');
 vm.runInContext(`
 document.getElementById('dataShape').value='blobs';
 document.getElementById('numPoints').value='50';
 drawData=()=>{};updateDataDisplay=()=>{};
 apiClient={generateData:()=>new Promise(resolve=>window.resolveDataset=resolve)};
 `,sandbox);
 const generating=vm.runInContext('generateData()',sandbox);
 vm.runInContext(`clearIterationHistory();window.resolveDataset({status:'success',session_key:'generated',data:{X:[[0,0],[1,1]]}})`,sandbox);
 await generating;
 assert.equal(vm.runInContext('currentSessionKey',sandbox),'generated','Changing algorithm must not cancel a pending dataset request');
 console.log('PASS: dataset generation survives algorithm changes');
})().catch(error=>{console.error(error);process.exitCode=1;});

vm.runInContext(`
window.finalColors=[];
drawDataPoint=(ctx,x,y,color)=>window.finalColors.push(color);
updateClusterDisplay=()=>{};
clusterCtx={}; clusterCanvas={}; clearCanvas=()=>{};
data=[{x:0,y:0},{x:.5,y:.5},{x:1,y:1}];
displayFinalResults([2,0,1]);
if (window.finalColors.join() !== [clusterColors[2],clusterColors[0],clusterColors[1]].join()) throw new Error('Final colors differ from replay cluster IDs');
`,sandbox);
console.log('PASS: final results preserve replay color IDs even when label order differs');
