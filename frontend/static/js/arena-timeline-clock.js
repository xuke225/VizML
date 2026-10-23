/* A pure measured-time clock, shared by the UI and deterministic unit tests. */
((host) => {
    'use strict';
    class ArenaTimelineClock {
        // Local estimates never train models or issue requests while typing.
        // Cold-start constants are deliberately rough; recorded wall times
        // calibrate prediction/encoding separately from the replay clock.
        static estimateRecording({steps, models, params, pointCount, trainCount, profiles={}}) {
            const sum = n => n * (n + 1) / 2;
            const tracks = models.map(key => {
                const p = params[key] || {}, profile = profiles[key];
                const independent = key === 'svm' || (key === 'mlp' && p.network_type !== 'perceptron' && p.solver === 'lbfgs');
                const treeLimit = Math.min(trainCount - 1, p.max_depth == null ? trainCount - 1 : 2 ** p.max_depth - 1);
                const limits = {svm:p.max_iter || 1000, mlp:p.max_iter || 500, forest:p.n_estimators || 10,
                    knn:p.n_neighbors || 5, tree:Math.max(1, treeLimit), bayes:Math.ceil(trainCount / 8)};
                const count = Math.min(steps, profile?.complete ? profile.frames.length : limits[key]);
                let ms;
                if (profile?.frames.length && Number.isFinite(profile.recording_ms)) {
                    const frames = profile.frames, n = frames.length;
                    const compute = frames.reduce((total, f) => total + f.duration_ms, 0);
                    const encoding = Math.max(0, profile.recording_ms - profile.setup_ms - compute) / n;
                    const cost = count <= n ? frames.slice(0,count).reduce((total,f)=>total+f.duration_ms,0) :
                        compute * (independent ? sum(count) / sum(n) : count / n);
                    ms = profile.setup_ms + cost + encoding * count;
                } else {
                    const scale = Math.max(0.15, trainCount / 135);
                    const layers = String(p.hidden_layer_sizes || '100,50').split(',').map(Number);
                    const widths = [2,...layers,2];
                    const weights = widths.slice(1).reduce((total,width,i)=>total+width*widths[i],0);
                    const complexity = p.network_type === 'perceptron' ? 0.1 : Math.max(0.15, weights / 5300);
                    // Includes the fixed 120×120 grid and per-point explanations.
                    const encoding = (key === 'forest' ? 24 : 12) * (0.5 + pointCount / 360);
                    const perStep = {svm:0.04*scale*scale, knn:0.08*scale, tree:0.03,
                        forest:3*scale, mlp:(independent ? 0.15 : 2)*scale*complexity, bayes:0.5*scale}[key];
                    ms = 3*scale + encoding*count + perStep*(independent ? sum(count) : count);
                }
                const calibrated = Boolean(profile);
                return {key, count, ms, calibrated, low:ms*(calibrated ? 0.6 : 0.25), high:ms*(calibrated ? 2 : 4)};
            });
            return {tracks, count:tracks.reduce((n,t)=>n+t.count,0),
                low:tracks.reduce((n,t)=>n+t.low,0), high:tracks.reduce((n,t)=>n+t.high,0),
                calibrated:tracks.filter(t=>t.calibrated).length};
        }
        constructor(tracks, speed=0.001) {
            this.tracks=tracks;
            this.duration=Math.max(0,...Object.values(tracks).map(t=>t.duration_ms));
            this.time=0;this.speed=speed;this.playing=false;this.last=null;
        }
        tick(now) {
            if (this.playing && this.last !== null) this.time=Math.min(this.duration,this.time+Math.max(0,now-this.last)*this.speed);
            this.last=now;
            if (this.time>=this.duration) this.playing=false;
            return this.positions();
        }
        play(now) {this.last=now;this.playing=this.time<this.duration;}
        pause(now) {this.tick(now);this.playing=false;}
        setSpeed(value,now) {
            if (!Number.isFinite(value) || value<0.00001 || value>10) throw new RangeError('倍率范围为 0.00001～10');
            this.tick(now);this.speed=value;
        }
        seek(time,now) {
            this.time=Math.max(0,Math.min(this.duration,time));this.last=now;
            if(this.time>=this.duration)this.playing=false;
            return this.positions();
        }
        positions() {
            return Object.fromEntries(Object.entries(this.tracks).map(([key,track])=>{
                let low=0,high=track.frames.length;
                while(low<high){const mid=(low+high)>>1;if(track.frames[mid].elapsed_ms<=this.time)low=mid+1;else high=mid;}
                return [key,low-1];
            }));
        }
    }
    if (typeof module !== 'undefined' && module.exports) module.exports=ArenaTimelineClock;
    else host.ArenaTimelineClock=ArenaTimelineClock;
})(typeof window !== 'undefined' ? window : globalThis);
