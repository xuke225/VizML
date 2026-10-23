const assert=require('node:assert/strict');
const {chromium}=require('playwright');
const path=require('node:path');
const fs=require('node:fs');
const output=path.resolve('.arena-qa');fs.mkdirSync(output,{recursive:true});
const base=process.env.ARENA_URL||'http://localhost:5432';

(async()=>{
    const browser=await chromium.launch({headless:true,executablePath:process.env.ARENA_BROWSER_EXE||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
    const page=await browser.newPage({viewport:{width:1821,height:866}});
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    const snap=()=>page.evaluate(()=>window.VizMLArena.snapshot());
    const train=async()=>{
        await page.locator('#arena-run').click();
        await page.waitForFunction(()=>{const s=window.VizMLArena.snapshot();return s.fresh&&!s.busy&&s.revealProgress===1;});
    };
    const choose=async selected=>{
        for(const key of selected)await page.locator('[data-arena-participant="'+key+'"]').check();
        for(const key of ['svm','knn','tree','forest','mlp','bayes'].filter(k=>!selected.includes(k)))await page.locator('[data-arena-participant="'+key+'"]').uncheck();
        assert.equal(await page.locator('.arena-model:visible').count(),selected.length);
    };
    const fits=async()=>{
        await page.locator('#arena-fit-canvases').click();
        await page.waitForFunction(()=>[...document.querySelectorAll('.arena-model:not([hidden]) canvas')].every(c=>{
            const r=c.getBoundingClientRect();return r.top>=0&&r.bottom<=innerHeight&&r.left>=0&&r.right<=innerWidth;
        }));
        assert.ok(await page.locator('.arena-model:visible').evaluateAll(es=>es.every(e=>{
            const r=e.getBoundingClientRect();return r.top>=0&&r.bottom<=innerHeight;
        })),'whole cards, including collapse controls, must fit');
        const stickyBottom=await page.locator('.arena-topline').evaluate(e=>e.getBoundingClientRect().bottom);
        assert.ok(await page.locator('.arena-model:visible canvas').evaluateAll((es,bottom)=>es.every(e=>e.getBoundingClientRect().top>=bottom),stickyBottom));
    };
    try{
        await page.goto(base+'/#ml-arena');
        await page.waitForFunction(()=>window.VizMLArena?.snapshot().split);
        assert.equal(await page.locator('.arena-model-data').count(),6);
        assert.equal(await page.locator('.arena-model-data[open]').count(),0);
        await train();
        const initial=await snap(), calls=initial.requestCount;
        await fits();
        await page.screenshot({path:path.join(output,'six-collapsed-1821x866.png')});
        await page.locator('#arena-model-data-svm>summary').click();
        assert.equal(await page.locator('#arena-score-svm').isVisible(),true);
        assert.equal(await page.locator('#arena-score-knn').isVisible(),false);
        await page.locator('#arena-model-data-svm>summary').press('Enter');
        assert.equal(await page.locator('#arena-score-svm').isVisible(),false);
        await page.locator('#arena-data-expand').click();
        assert.equal(await page.locator('.arena-model-score:visible').count(),6);
        const grids=(await snap()).result.models;
        await page.locator('#arena-data-collapse').click();
        await fits();
        assert.deepEqual((await snap()).result.models,grids);
        assert.equal((await snap()).requestCount,calls);
        console.log('PASS per-model native keyboard folding, expand/collapse all, no retraining or data loss');

        for(const [width,height] of [[1920,1080],[1440,900],[1366,768],[1280,720],[900,768]]){
            await page.setViewportSize({width,height});await fits();
            assert.equal(await page.locator('.arena-model:visible').count(),6);
        }
        await page.setViewportSize({width:1440,height:900});
        await page.locator('#arena-focus').click();await fits();
        await page.screenshot({path:path.join(output,'six-collapsed-focus.png')});
        await page.keyboard.press('Escape');
        console.log('PASS six full canvases/cards simultaneously fit desktop, laptop, tablet and focused view');

        await choose(['svm','knn']);
        assert.equal((await snap()).fresh,false);
        assert.equal(await page.locator('[data-arena-participant=svm]').isDisabled(),true);
        assert.equal(await page.locator('[data-arena-participant=knn]').isDisabled(),true);
        assert.deepEqual(Object.keys((await snap()).parameters),['svm','knn']);
        await train();let two=await snap();
        assert.deepEqual(two.result.model_keys,['svm','knn']);
        assert.deepEqual(Object.keys(two.result.models).sort(),['knn','svm']);
        assert.equal(Object.keys(two.result.ranks).length,2);
        assert.deepEqual(two.split,initial.split);
        assert.equal(await page.locator('.arena-models').evaluate(e=>getComputedStyle(e).gridTemplateColumns.split(' ').length),2);
        await fits();
        await page.screenshot({path:path.join(output,'two-selected-collapsed.png')});
        await page.locator('.arena-metrics>summary').click();
        assert.equal(await page.locator('#arena-metrics-body tr').count(),2);
        // A second same-lineup run exercises baseline comparison with only 2 keys.
        await train();assert.ok((await page.locator('#arena-compare-note').textContent()).includes('KNN'));
        await page.locator('[data-observe-step="10"]').click();
        await page.waitForFunction(()=>{const s=window.VizMLArena.snapshot();return !s.busy&&s.result?.observation?.step===10;});
        assert.deepEqual(Object.keys((await snap()).result.models).sort(),['knn','svm']);
        console.log('PASS two-model actual training, ranking, same holdout, comparison and single-step observation');

        await choose(['tree','forest','mlp']);await train();
        assert.deepEqual((await snap()).result.model_keys,['tree','forest','mlp']);
        await fits();
        await choose(['tree','forest','mlp','bayes']);await train();
        assert.equal(await page.locator('.arena-models').evaluate(e=>getComputedStyle(e).gridTemplateColumns.split(' ').length),2);
        await fits();
        await choose(['svm','knn','tree','forest','mlp']);await train();await fits();
        await page.locator('#arena-participants-all').click();await train();await fits();
        assert.equal(Object.keys((await snap()).result.models).length,6);
        console.log('PASS 3/4/5/6-model rosters and balanced responsive grids');

        await choose(['forest','mlp']);
        await page.locator('#arena-play-steps').fill('4');await page.locator('#arena-play-speed').fill('0.00001');
        await page.locator('#arena-play-record').click();
        await page.waitForFunction(()=>window.VizMLArena.snapshot().playback?.active);
        await page.locator('#arena-play-pause').click();
        const recording=await snap();
        assert.deepEqual(Object.keys(recording.playback.tracks),['forest','mlp']);
        const time=recording.playback.duration;
        await page.locator('#arena-play-seek').evaluate((el,time)=>{el.value=time;el.dispatchEvent(new Event('input',{bubbles:true}));},time);
        await fits();
        assert.equal(await page.locator('.arena-process:visible').count(),0);
        const before=(await snap()).requestCount;
        await page.locator('#arena-data-expand').click();
        assert.equal(await page.locator('.arena-process:visible').count(),2);
        assert.equal(await page.locator('#arena-vote-forest').isVisible(),true);
        assert.equal((await snap()).result.models.mlp.fit_info.iterations,4);
        await page.locator('#arena-data-collapse').click();await fits();
        assert.equal((await snap()).requestCount,before);
        await choose(['svm','knn']);
        assert.equal((await snap()).playback,null);
        assert.equal((await snap()).fresh,false);
        console.log('PASS selected-only continuous replay, live hidden data, collapse and lineup invalidation');

        await page.setViewportSize({width:390,height:844});
        await page.locator('#arena-data-expand').click();
        assert.equal(await page.locator('.arena-model-score:visible').count(),2);
        assert.ok(await page.locator('#ml-arena').evaluate(e=>e.scrollWidth<=e.clientWidth+1));
        assert.equal(await page.locator('#arena-fit-canvases').textContent(),'定位模型画布 ↓');
        assert.deepEqual(errors,[]);
        console.log('PASS mobile folding and no JavaScript errors');
        console.log('ALL LAYOUT / SELECTION BROWSER CHECKS PASSED');
    }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
