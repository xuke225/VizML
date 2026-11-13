const assert=require('node:assert/strict');
const path=require('node:path');
const fs=require('node:fs');
const {chromium}=require('playwright');
const output=path.resolve('.arena-qa');fs.mkdirSync(output,{recursive:true});
const base=process.env.ARENA_URL || 'http://localhost:5432';

(async()=>{
    const browser=await chromium.launch({headless:true,executablePath:process.env.ARENA_BROWSER_EXE || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
    const page=await browser.newPage({viewport:{width:1440,height:1100}});
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    const snap=()=>page.evaluate(()=>window.VizMLArena.snapshot());
    const seek=async time=>page.locator('#arena-play-seek').evaluate((el,time)=>{el.value=time;el.dispatchEvent(new Event('input',{bubbles:true}));},time);
    const speed=async value=>page.locator('#arena-play-speed').fill(String(value));
    try {
        await page.goto(base+'/#ml-arena');
        await page.waitForFunction(()=>window.VizMLArena?.snapshot().split);
        await page.locator('#arena-run').click();
        await page.waitForFunction(()=>{const s=window.VizMLArena.snapshot();return s.fresh&&!s.busy&&s.revealProgress===1;});
        const final=(await snap()).result;
        await page.locator('#arena-play-steps').fill('12');await speed(0.00001);
        const response=page.waitForResponse(r=>r.url().endsWith('/api/arena/timeline')&&r.request().method()==='POST');
        await page.locator('#arena-play-record').click();
        const record=await (await response).json();
        await page.waitForFunction(()=>{const s=window.VizMLArena.snapshot();return s.playback?.active&&!s.busy;});
        await page.locator('#arena-play-pause').click();
        let state=await snap();
        assert.equal(state.playback.playing,false);
        assert.equal(await page.locator('#arena-step-next').isDisabled(),true);
        const frozen=state.playback.time,calls=state.requestCount;
        await page.waitForTimeout(200);
        assert.equal((await snap()).playback.time,frozen);
        assert.equal(await page.locator('#arena-baseline').isDisabled(),true);
        assert.equal(await page.locator('[data-rank]').count(),0);
        assert.deepEqual(state.result.ranks,{});
        console.log('PASS record/play/pause: six independent measured tracks, no ranking, exact freeze');

        const firstTimes=Object.values(record.tracks).map(t=>t.frames[0].elapsed_ms).sort((a,b)=>a-b);
        await seek((firstTimes[0]+firstTimes[1])/2);
        state=await snap();
        assert.ok(Object.values(state.result.models).some(m=>m===null));
        assert.ok(Object.values(state.result.models).some(m=>m!==null));
        assert.equal(state.playback.playing,false);
        // Independent tracks must select the latest completed, not next, snapshot.
        const time=record.tracks.mlp.frames[5].elapsed_ms+0.000001;
        await seek(time);
        state=await snap();
        for(const [key,track] of Object.entries(record.tracks)){
            const index=track.frames.reduce((last,f,i)=>f.elapsed_ms<=state.playback.time?i:last,-1);
            assert.equal(state.playback.positions[key],index);
            if(index>=0){
                const bytes=Buffer.from(track.frames[index].model.grid_bits,'base64');
                const expected=Array.from({length:14400},(_,i)=>(bytes[i>>3]>>(7-(i&7)))&1);
                assert.deepEqual(state.result.models[key].grid,expected);
            }
        }
        assert.equal(state.result.models.mlp.fit_info.iterations,6);
        assert.equal(state.requestCount,calls,'seeking must not retrain');
        await page.locator('#ml-arena').screenshot({path:path.join(output,'timeline-desktop.png')});
        console.log('PASS actual time axis, independent frame positions, exact packed boundaries and real epochs');

        const pausedTime=state.playback.time;
        await speed(0.00002);
        assert.equal((await snap()).playback.time,pausedTime);
        await page.locator('#arena-play-pause').click();
        await page.waitForTimeout(120);
        const moving=await snap();
        assert.ok(moving.playback.time>pausedTime);
        await speed(0.00004);
        assert.equal((await snap()).playback.speed,0.00004);
        assert.equal((await snap()).requestCount,calls);
        await page.locator('#arena-play-pause').click();
        const afterPause=(await snap()).playback.time;
        await page.waitForTimeout(150);
        assert.equal((await snap()).playback.time,afterPause);
        // A background-tab transition pauses instead of catching up on return.
        await page.locator('#arena-play-pause').click();
        await page.evaluate(()=>{
            Object.defineProperty(document,'hidden',{configurable:true,get:()=>true});
            document.dispatchEvent(new Event('visibilitychange'));
            delete document.hidden;
        });
        assert.equal((await snap()).playback.playing,false);
        // Slider uses the same live multiplier and supports immediate updates.
        const backgroundPaused=(await snap()).playback.time;
        await page.locator('#arena-play-speed-slider').evaluate(el=>{el.value='-3';el.dispatchEvent(new Event('input',{bubbles:true}));});
        assert.equal((await snap()).playback.speed,0.001);
        assert.ok((await page.locator('#arena-play-speed-note').innerText()).includes('1000'));
        assert.equal((await snap()).playback.time,backgroundPaused);
        console.log('PASS pause/resume and live numeric/slider speed edits without resets or HTTP calls');

        await speed(0.00001);await page.locator('#arena-play-restart').click();
        await page.locator('#arena-play-pause').click();
        assert.ok((await snap()).playback.time<afterPause);
        assert.equal((await snap()).requestCount,calls);
        await seek(record.timeline.duration_ms);
        assert.equal((await snap()).playback.playing,false);
        assert.ok((await page.locator('#arena-play-state').innerText()).includes('本段演示结束'));
        await page.locator('#arena-step-final').click();
        state=await snap();
        assert.equal(state.playback.active,false);
        assert.equal(state.result.observation,null);
        assert.deepEqual(state.result.models.mlp.grid,final.models.mlp.grid);
        console.log('PASS replay from start, finish and untouched complete fit/baseline');

        await page.locator('#arena-play-restart').click();
        await page.locator('#arena-play-pause').click();
        await page.setViewportSize({width:390,height:844});
        await seek(record.tracks.mlp.frames[5].elapsed_ms);
        assert.ok(await page.locator('#ml-arena').evaluate(el=>el.scrollWidth<=el.clientWidth+1));
        assert.ok(await page.locator('.arena-playback').evaluate(el=>el.scrollWidth<=el.clientWidth+1));
        assert.equal(await page.locator('.arena-model canvas').count(),6);
        await page.locator('#ml-arena').screenshot({path:path.join(output,'timeline-mobile.png')});
        await page.locator('#arena-test-size').selectOption('0.3');
        state=await snap();
        assert.equal(state.playback,null);
        assert.equal(state.fresh,false);
        assert.equal(await page.locator('#arena-play-pause').isDisabled(),true);
        assert.equal(await page.locator('#arena-play-restart').isDisabled(),true);
        console.log('PASS responsive layout and invalidation after data/split changes');

        // A cancel or edit must not allow a late recording to replace the view.
        let started;
        const requested=new Promise(resolve=>started=resolve);
        await page.route('**/api/arena/timeline',async route=>{
            const response=await route.fetch();started();await new Promise(resolve=>setTimeout(resolve,400));
            try{await route.fulfill({response});}catch(_){}
        });
        await page.locator('#arena-play-steps').fill('2');
        await page.locator('#arena-play-record').click();await requested;
        await page.locator('#arena-play-cancel').click();await page.waitForTimeout(500);
        assert.equal((await snap()).recording,false);
        assert.equal((await snap()).busy,false);
        assert.equal((await snap()).playback,null);
        await page.unroute('**/api/arena/timeline');
        assert.deepEqual(errors,[]);
        console.log('PASS cancel recording/late-response guard, no JavaScript errors');
        console.log('ALL TIMELINE BROWSER CHECKS PASSED');
    }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
