/* Run against the local Flask server. Uses an existing Playwright installation. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');
const output = path.resolve('.arena-qa');
const base = process.env.ARENA_URL || 'http://localhost:5432';
fs.mkdirSync(output, {recursive:true});

(async () => {
    const browser = await chromium.launch({headless:true,
        executablePath:process.env.ARENA_BROWSER_EXE || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
    const page = await browser.newPage({viewport:{width:1440,height:1100},deviceScaleFactor:1});
    const errors = [];
    page.on('pageerror',error => errors.push(error.message));
    const snap = () => page.evaluate(() => window.VizMLArena.snapshot());
    const fresh = () => page.waitForFunction(() => {
        const s=window.VizMLArena.snapshot();return s.fresh && !s.busy && s.revealProgress===1;
    });
    const train = async () => { await page.locator('#arena-run').click(); await fresh(); };
    const pos = async (canvas,point) => {
        const box=await page.locator(canvas).boundingBox();
        return {x:box.x+25+(point.x+3)/6*(box.width-37),y:box.y+12+(3-point.y)/6*(box.height-34)};
    };
    const screenshot = async name => {
        await page.addStyleTag({content:'.site-header,.skip-link{visibility:hidden!important}'});
        await page.locator('#ml-arena').screenshot({path:path.join(output,name)});
        await page.addStyleTag({content:'.site-header,.skip-link{visibility:visible!important}'});
    };
    try {
        await page.goto(base+'/#ml-arena');
        await page.waitForFunction(() => window.VizMLArena?.snapshot().split);
        assert.equal((await snap()).points.length,180);
        assert.equal(await page.locator('#ai-tech-forge').count(),0);
        assert.equal(await page.locator('.lab-card').count(),11);
        await train();
        let initial=await snap();
        assert.deepEqual(initial.result.model_keys,['svm','knn','tree','forest','mlp','bayes']);
        assert.equal(await page.locator('.arena-model canvas').count(),6);
        assert.equal(await page.locator('.arena-model select').count(),0);
        const ranges=await page.locator('.ml-arena canvas').evaluateAll(cs => cs.map(c => c.dataset.bounds));
        assert.ok(ranges.every(r => r==='-3,3,-3,3'));
        for(const model of Object.values(initial.result.models)) {
            assert.equal(model.grid.length,14400);
            assert.ok(new Set(model.grid).size===2);
            assert.deepEqual(model.test_ids,initial.split.test_ids);
        }
        console.log('PASS core: generation, six real models, six permanent plots, shared split');
        await page.setViewportSize({width:1440,height:900});
        await page.locator('#arena-focus').click();
        await page.locator('#arena-fit-canvases').click();
        assert.ok(await page.locator('.arena-model-plot').first().evaluate(e=>e.getBoundingClientRect().bottom<=innerHeight));
        assert.equal(await page.locator('.arena-models').evaluate(e=>getComputedStyle(e).gridTemplateColumns.split(' ').length),3);
        await page.screenshot({path:path.join(output,'05-focus-900.png')});
        await page.keyboard.press('Escape');
        await page.setViewportSize({width:1440,height:1100});
        await page.waitForFunction(() => document.querySelector('#arena-plot-tree').dataset.boundaryVersion===document.querySelector('#ml-arena').dataset.version);
        await screenshot('01-arena-desktop.png');
        // Hover cannot train, but all seven plots must highlight the same ID.
        const trainPoint=initial.points.find(p => p.role==='train');
        const from=await pos('#arena-data',trainPoint);
        const requestCount=initial.requestCount;
        await page.mouse.move(from.x,from.y);
        await page.waitForFunction(() => new Set([...document.querySelectorAll('.ml-arena canvas')].map(c=>c.dataset.highlight)).size===1);
        assert.equal((await snap()).hover,trainPoint.id);
        assert.equal((await snap()).requestCount,requestCount);
        // Drag a real training point, then compare actual returned grids.
        await page.mouse.down();
        await page.mouse.move(from.x+85,from.y-50,{steps:12});
        await page.mouse.up();
        await page.waitForFunction(() => !window.VizMLArena.snapshot().fresh);
        assert.ok(await page.locator('.arena-stale-tag').first().isVisible());
        await page.waitForFunction(() => document.querySelector('#arena-run').disabled===false);
        await train();
        let moved=await snap();
        assert.deepEqual(moved.split,initial.split);
        assert.equal(moved.result.holdout_fingerprint,initial.result.holdout_fingerprint);
        assert.ok(['svm','knn','tree'].some(k => JSON.stringify(initial.result.models[k].grid)!==JSON.stringify(moved.result.models[k].grid)));
        console.log('PASS drag: stale immediately, same holdout, boundaries recomputed');
        const wrong=moved.points.find(p => p.role==='test' && moved.result.models.tree.predictions[p.id]!==p.clean_label);
        assert.ok(wrong,'default data must include a tree error for inspection');
        await page.locator('[data-tool=inspect]').click();
        const click=await pos('#arena-data',wrong);
        await page.mouse.click(click.x,click.y);
        assert.equal((await snap()).selected,wrong.id);
        assert.match(await page.locator('#arena-explain-tree').textContent(),/误判/);
        assert.match(await page.locator('#arena-explain-knn').textContent(),/真实训练邻居/);
        assert.match(await page.locator('#arena-explain-svm').textContent(),/决策分数/);
        assert.match(await page.locator('#arena-explain-forest').textContent(),/棵树概率的平均/);
        assert.match(await page.locator('#arena-explain-mlp').textContent(),/adam 实际/);
        assert.match(await page.locator('#arena-explain-bayes').textContent(),/高斯条件独立/);
        console.log('PASS inspection: test errors and true mechanism explanations');
        // Challenged preview and fit must both leave the holdout untouched.
        await page.locator('#arena-baseline').click();
        const beforeChallenge=await snap();
        await page.evaluate(() => {
            for(const [id,value] of [['arena-challenge-noise','0.35'],['arena-corruption','0.2']]) {
                const slider=document.getElementById(id);slider.value=value;
                slider.dispatchEvent(new Event('input',{bubbles:true}));
                slider.dispatchEvent(new Event('change',{bubbles:true}));
            }
        });
        await page.waitForFunction(() => window.VizMLArena.snapshot().points.some(p=>p.corrupted));
        assert.equal((await snap()).fresh,false);
        const oldTest=beforeChallenge.points.filter(p=>p.role==='test');
        assert.deepEqual((await snap()).points.filter(p=>p.role==='test'),oldTest);
        assert.equal(Object.keys((await snap()).result.models).length,6);
        assert.ok((await snap()).points.filter(p=>p.corrupted).every(p=>p.role==='train'));
        await train();
        assert.deepEqual((await snap()).points.filter(p=>p.role==='test'),oldTest);
        await page.locator('#arena-compare-enabled').check();
        await page.locator('#arena-divider').fill('58');
        await screenshot('02-challenge-comparison.png');
        console.log('PASS challenges: train-only noise/corruption and genuine split comparison');
        // Simulate an old response arriving after a new experiment.
        await page.locator('#arena-reset-challenge').click();
        await page.waitForFunction(() => !window.VizMLArena.snapshot().points.some(p=>p.corrupted));
        let releaseOld, sawOld;
        const held=new Promise(r=>releaseOld=r), observed=new Promise(r=>sawOld=r);
        let once=true;
        await page.route('**/api/arena/run',async route=>{
            if(!once){await route.continue();return;}
            once=false;
            const response=await route.fetch();
            sawOld();
            await held;
            try{await route.fulfill({response});}catch(_){}
        });
        await page.locator('#arena-run').click();
        await observed;
        await page.locator('#arena-c').fill('8');
        await page.locator('#arena-c').press('Tab');
        const newVersion=(await snap()).version;
        await train();
        releaseOld();
        await page.waitForFunction(v=>window.VizMLArena.snapshot().resultVersion===v,newVersion);
        assert.equal((await snap()).fresh,true);
        await page.unroute('**/api/arena/run');
        console.log('PASS concurrency: stale response cannot overwrite newer data');
        // New parameters invalidate all results; nonconvergence is reported honestly.
        await page.locator('.arena-params details').filter({has:page.locator('#arena-mlp-iterations')}).locator('summary').click();
        await page.locator('#arena-mlp-hidden').selectOption('custom');
        await page.locator('#arena-mlp-hidden-custom').fill('4');
        await page.locator('#arena-mlp-iterations').fill('100');
        await page.locator('#arena-mlp-iterations').press('Tab');
        assert.equal((await snap()).fresh,false);
        await train();
        assert.equal((await snap()).result.models.mlp.fit_info.iteration_limit_reached,true);
        assert.match(await page.locator('#arena-explain-mlp').textContent(),/未满足停止条件/);
        console.log('PASS MLP: real iteration count, honest limit warning, all six rerun');
        // Undo, clear, free draw, deleting and relabeling should remain real data edits.
        await page.locator('#arena-preset').selectOption('free');
        await page.waitForFunction(()=>window.VizMLArena.snapshot().points.length===0);
        assert.equal(await page.locator('#arena-run').isDisabled(),true);
        await page.locator('#arena-data').scrollIntoViewIfNeeded();
        for(const [tool,row] of [['a',0],['b',1]]) {
            await page.locator('[data-tool='+tool+']').click();
            const box=await page.locator('#arena-data').boundingBox();
            for(let i=0;i<14;i++) await page.mouse.click(box.x+45+i*19,box.y+55+row*110+(i%3)*12);
        }
        await page.waitForFunction(()=>window.VizMLArena.snapshot().points.length===28);
        await train();
        let drawn=await snap();
        assert.equal(drawn.points.length,28);
        await page.locator('[data-tool=flip]').click();
        let at=await pos('#arena-data',drawn.points[0]);
        await page.mouse.click(at.x,at.y);
        assert.equal((await snap()).base.find(p=>p.id===drawn.points[0].id).label,1-drawn.points[0].label);
        await page.locator('#arena-undo').click();
        await page.waitForFunction(()=>window.VizMLArena.snapshot().base.filter(p=>p.label===0).length===14);
        await page.locator('[data-tool=erase]').click();
        at=await pos('#arena-data',(await snap()).points[0]);
        await page.mouse.click(at.x,at.y);
        assert.equal((await snap()).points.length,27);
        await page.locator('#arena-clear').click();
        assert.equal((await snap()).points.length,0);
        console.log('PASS editing: free draw, label flip, delete, clear and undo');
        await page.setViewportSize({width:1024,height:900});
        await page.locator('#arena-preset').selectOption('xor');
        await page.waitForFunction(()=>window.VizMLArena.snapshot().points.length===180);
        await train();
        assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
        assert.equal(await page.locator('.arena-models').evaluate(e=>getComputedStyle(e).gridTemplateColumns.split(' ').length),2);
        await screenshot('03-arena-tablet.png');
        const mobile=await browser.newContext({viewport:{width:390,height:844},isMobile:true,hasTouch:true});
        const touch=await mobile.newPage();
        touch.on('pageerror',error=>errors.push(error.message));
        await touch.goto(base+'/#ml-arena');
        await touch.waitForFunction(()=>window.VizMLArena?.snapshot().split);
        await touch.locator('#arena-preset').selectOption('circles');
        await touch.waitForFunction(()=>window.VizMLArena.snapshot().points.length===180);
        await touch.locator('#arena-run').tap();
        await touch.waitForFunction(()=>{
            const s=window.VizMLArena.snapshot();return s.fresh && !s.busy && s.revealProgress===1;
        });
        assert.equal(await touch.locator('.arena-model canvas').count(),6);
        assert.ok(await touch.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
        await touch.addStyleTag({content:'.site-header,.skip-link{visibility:hidden!important}'});
        await touch.locator('#ml-arena').screenshot({path:path.join(output,'04-arena-mobile.png')});
        await mobile.close();
        assert.deepEqual(errors,[]);
        console.log('PASS responsive: tablet layout, mobile touch generation/training, no JS errors');
        console.log('ALL ARENA BROWSER CHECKS PASSED');
    } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
