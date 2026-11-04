/* Real checkpoint integration, local server required. */
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const {chromium} = require('playwright');
const base = process.env.ARENA_URL || 'http://localhost:5432';
const output = path.resolve('.arena-qa');
fs.mkdirSync(output,{recursive:true});

(async()=>{
    const browser = await chromium.launch({headless:true, executablePath:process.env.ARENA_BROWSER_EXE || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
    const page = await browser.newPage({viewport:{width:1440,height:1100}});
    const errors=[];
    page.on('pageerror',error=>errors.push(error.message));
    const snap=()=>page.evaluate(()=>window.VizMLArena.snapshot());
    const done=step=>page.waitForFunction(step=>{
        const s=window.VizMLArena.snapshot();
        return s.fresh && !s.busy && s.result?.observation?.step===step;
    },step);
    try {
        await page.goto(base+'/#ml-arena');
        await page.waitForFunction(()=>window.VizMLArena?.snapshot().split);
        await page.locator('#arena-data-expand').click();
        // Observation works without a final fit or a baseline.
        await page.locator('[data-observe-step="1"]').click();await done(1);
        let first=await snap();
        assert.equal(Object.keys(first.result.models).length,6);
        assert.equal(await page.locator('.arena-process:visible').count(),6);
        assert.deepEqual(first.result.ranks,{});
        assert.equal(await page.locator('#arena-baseline').isDisabled(),true);
        assert.equal(await page.locator('#arena-step-final').isDisabled(),true);
        assert.equal(first.result.models.mlp.fit_info.iterations,1);
        assert.equal(first.result.models.mlp.fit_info.loss_curve.length,1);
        assert.equal(first.result.models.svm.effective_params.max_iter,1);
        assert.equal(first.result.models.forest.fit_info.tree_count,1);
        assert.equal(first.result.models.bayes.observation.seen_count,8);
        assert.ok((await page.locator('#arena-explain-knn').innerText()).includes('部分投票'));
        assert.equal(await page.locator('[data-rank]').count(),0);
        console.log('PASS first checkpoint without baseline; all six real processes, no ranking');

        await page.locator('[data-observe-step="10"]').click();await done(10);
        const ten=await snap();
        assert.deepEqual(ten.split,first.split);
        assert.equal(ten.result.models.mlp.fit_info.iterations,10);
        assert.equal(ten.result.models.mlp.fit_info.loss_curve.length,10);
        assert.equal(ten.result.models.forest.fit_info.tree_count,10);
        assert.equal(ten.result.models.knn.observation.current,5);
        assert.deepEqual(ten.result.models.knn.grid,first.result.models.knn.grid);
        assert.notDeepEqual(ten.result.models.mlp.grid,first.result.models.mlp.grid);
        assert.ok(await page.locator('#arena-process-mlp svg').count());
        await page.locator('#ml-arena').screenshot({path:path.join(output,'observation-10-desktop.png')});
        const calls=ten.requestCount;
        await page.locator('#arena-step-history').selectOption('1');await done(1);
        assert.equal((await snap()).requestCount,calls);
        assert.deepEqual((await snap()).result.models.mlp.grid,first.result.models.mlp.grid);
        // Clicking a different query changes all explanations, without retraining.
        await page.locator('#arena-plot-knn').focus();await page.keyboard.press('ArrowRight');
        assert.notEqual((await snap()).selected,first.selected);
        assert.equal((await snap()).requestCount,calls);
        await page.locator('#arena-step-next').click();await done(2);
        await page.locator('#arena-step-ten').click();await done(12);
        console.log('PASS step 1/10, +1/+10, cached rewind and local KNN query inspection');

        await page.locator('#arena-run').click();
        await page.waitForFunction(()=>{const s=window.VizMLArena.snapshot();return s.fresh&&!s.busy&&!s.result.observation&&s.revealProgress===1;});
        const final=await snap();
        assert.equal(Object.keys(final.result.ranks).length,6);
        assert.equal(await page.locator('.arena-process:visible').count(),0);
        await page.locator('[data-observe-step="1"]').click();await done(1);
        assert.equal(await page.locator('#arena-step-final').isDisabled(),false);
        await page.locator('#arena-step-final').click();
        assert.deepEqual((await snap()).result.models.mlp.grid,final.result.models.mlp.grid);
        assert.equal((await snap()).result.observation,null);
        assert.equal(await page.locator('#arena-baseline').isDisabled(),false);
        console.log('PASS full fit, rankings and baseline preserved independently of checkpoints');

        await page.locator('[data-observe-step="10"]').click();await done(10);
        // Invalid custom step does not issue a request.
        await page.locator('#arena-step-value').fill('0');
        const beforeInvalid=(await snap()).requestCount;
        await page.locator('#arena-step-go').click();
        assert.equal((await snap()).requestCount,beforeInvalid);
        assert.ok((await page.locator('#arena-status').innerText()).includes('整数'));
        await page.locator('#arena-c').fill('2');
        assert.equal((await snap()).fresh,false);
        assert.deepEqual((await snap()).checkpoints,[]);
        assert.equal(await page.locator('#arena-step-history').isDisabled(),true);
        assert.equal(await page.locator('#arena-step-final').isDisabled(),true);
        await page.locator('[data-observe-step="1"]').click();await done(1);
        assert.equal((await snap()).result.models.svm.effective_params.C,2);
        console.log('PASS invalid input rejected, parameter edits invalidate all old checkpoints');

        // An old observation response may not overwrite a newer data version.
        let started;
        const requested=new Promise(resolve=>started=resolve);
        await page.route('**/api/arena/observe',async route=>{
            const response=await route.fetch();started();
            await new Promise(resolve=>setTimeout(resolve,400));
            try {await route.fulfill({response});} catch (_) { /* intentionally aborted */ }
        });
        await page.locator('[data-observe-step="50"]').click();await requested;
        await page.locator('#arena-test-size').selectOption('0.3');
        await page.waitForFunction(()=>!window.VizMLArena.snapshot().busy);
        await page.waitForTimeout(500);
        const edited=await snap();
        assert.equal(edited.fresh,false);
        assert.deepEqual(edited.checkpoints,[]);
        assert.notEqual(edited.result.observation.step,50);
        await page.unroute('**/api/arena/observe');
        console.log('PASS delayed process response cannot overwrite edits');

        await page.setViewportSize({width:390,height:844});
        await page.locator('[data-observe-step="10"]').click();await done(10);
        assert.equal(await page.locator('.arena-process:visible').count(),6);
        assert.ok(await page.locator('#ml-arena').evaluate(el=>el.scrollWidth<=el.clientWidth+1));
        assert.ok(await page.locator('.arena-stepper').evaluate(el=>el.scrollWidth<=el.clientWidth+1));
        await page.locator('#ml-arena').screenshot({path:path.join(output,'observation-10-mobile.png')});
        assert.deepEqual(errors,[]);
        console.log('PASS mobile six-process layout; no JavaScript errors');
        console.log('ALL OBSERVATION BROWSER CHECKS PASSED');
    } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
