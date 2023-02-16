const lighthouse = require('lighthouse');
const Analyze = require('./chrome/wprofx/analyze');
const puppeteer = require('puppeteer');
const PuppeteerHar = require('puppeteer-har');
const argparse = require('argparse');
const Path = require('path');
const fs = require('fs');
const url = require('url');
const chromeHar = require('chrome-har');

const TRACE_CATEGORIES = [
    '-*',
    'toplevel',
    'blink.console',
    'disabled-by-default-devtools.timeline',
    'devtools.timeline',
    'disabled-by-default-devtools.timeline.frame',
    'devtools.timeline.frame',
    'disabled-by-default-devtools.timeline.stack',
    'disabled-by-default-v8.cpu_profile',
    'disabled-by-default-blink.feature_usage',
    'blink.user_timing',
    'v8.execute',
    'netlog',
];

const LIGHTHOUSE_CATEGORIES = [
    'first-contentful-paint',
    'first-meaningful-paint',
    'largest-contentful-paint',
    'speed-index',
    'interactive',
];
const ENDPOINTS = JSON.parse(fs.readFileSync(Path.join(__dirname,'endpoints.json'), 'utf8'));
const CONFIG = JSON.parse(fs.readFileSync(Path.join(__dirname, 'config.json'), 'utf8'));

const RETRIES = 5;
const ITERATIONS = CONFIG.iterations.value;

const DATA_PATH = Path.join(__dirname, CONFIG.data_path.value);
const TMP_DIR = Path.join(DATA_PATH, 'tmp');
const TIMINGS_DIR = Path.join(DATA_PATH, 'timings');
const NETLOG_DIR = Path.join(DATA_PATH, 'netlog');
const METRICS_DIR = Path.join(DATA_PATH, 'metrics');
const WPROFX_DIR = Path.join(DATA_PATH, 'wprofx');
const IMAGE_DIR = Path.join(DATA_PATH, 'images');
const DIRS = {
    'tmp': TMP_DIR,
    'timings': TIMINGS_DIR,
    'netlog': NETLOG_DIR,
    'metrics': METRICS_DIR,
    'wprofx': WPROFX_DIR,
    'images': IMAGE_DIR
}

//创建文件夹系--data
Object.values(DIRS).forEach((dir) => {
    fs.mkdirSync(dir, { recursive: true });
})

const DOMAINS = CONFIG.domains.value;
const SINGLE_SIZES = CONFIG.sizes.single;
const MULTI_SIZES = CONFIG.sizes.multi;

const CHROME_PROFILE = Path.join(TMP_DIR, 'chrome-profile');
const TMP_NETLOG = Path.join(TMP_DIR, 'chrome.json');

//////////////////////////////////////////////////////
const chromeArgs = (urls, log) => {
    const args = [
        // '--no-sandbox',
        '--headless',
        '--disable-gpu',
        '--disable-dev-shm-usage',
        // '--window-size=1920,1080',
        `--user-data-dir=${CHROME_PROFILE}`,
        '--disk-cache-dir=/dev/null',
        '--disk-cache-size=1',
        '--aggressive-cache-discard',
    ];

    if (log) {
        args.push(`--log-net-log=${TMP_NETLOG}`);
    }

    if (urls !== null && urls.length > 0) {
        args.push(
            '--enable-quic',
            '--quic-version=h3-29',
        );

        const origins = new Set();
        urls.forEach((urlString) => {
            const urlObject = url.parse(urlString);
            let port = '443';
            if (urlObject.port !== null) {
                port = urlObject.port;
            }
            origins.add(`${urlObject.host.split(':')[0]}:${port}`);
        });
        args.push(`--origin-to-force-quic-on=${Array.from(origins).join(',')}`);
    } else {
        args.push(
            '--disable-quic',
        );
    }

    return args;
};

//////////////////////////////////////////递归删除文件夹
const deleteFolderRecursive = (path) => {
    if (fs.existsSync(path)) {
        fs.readdirSync(path).forEach((file) => {
            const curPath = Path.join(path, file);
            if (fs.lstatSync(curPath).isDirectory()) { // recurse
                deleteFolderRecursive(curPath);
            } else { // delete file
                fs.unlinkSync(curPath);
            }
        });
        fs.rmdirSync(path);
    }
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
//////////////////////////////////////////////////////
const hasAltSvc = (entry) => {
    const { headers } = entry.response;
    for (const header of headers) {
        if (header.name === 'alt-svc' && header.value.includes('h3-29')) {
            return true;
        }
    }
    return false;
};

//////////////////////////////////////////////某某排序
const argsort = (array) => {
    const arrayObject = array.map((value, idx) => ({ value, idx }));
    arrayObject.sort((a, b) => {
        if (a.value < b.value) {
            return -1;
        }
        if (a.value > b.value) {
            return 1;
        }
        return 0;
    });
    return arrayObject.map((data) => data.idx);
};

///--------------------------------------------------
const runChromeWeb = async (urlObj, timings, file, netlogDir, wprofxDir, imageDir, isH3, log) => {
    const { url: urlString, size } = urlObj;
    const domains = [urlString];
    const prevLength = timings['speed-index'].length;

    //console.log("domains, prevLength: ", domains, prevLength);
    console.log("----------------------runChromeWeb Start.-----------------------");
    console.log("urlString in domains: ", `${urlString}`);

    //console.log("ITERATIONS: ", ITERATIONS);
    
    for(let i = prevLength; i < ITERATIONS; i += 1)
    {
        console.log(`--------------------------Iteration: ${i}`);

        for (let j = 0; j < RETRIES; j += 1)
        {
            const wprofx = new Analyze();

            try{
                //console.log("CHROME_PROFILE: ",CHROME_PROFILE);
                deleteFolderRecursive(CHROME_PROFILE);
                const args = chromeArgs(isH3 ? domains: null, log);
                args.push('--no-sandbox');
                // args.push('--ignoreHTTPSErrors');
                // args.push('--allow-insecure-localhost');//自建证书必加该行，本地服务器使用
                // ignoreHTTPSErrors
                //console.log("args: ", args);

                const browser = await puppeteer.launch({
                    headless: true,
                    defaultViewport: null,
                    args,
                });

                try{
                    const {lhr:{ audits }, artifacts, report} = await lighthouse(
                        urlString,
                        {
                            port: (new URL(browser.wsEndpoint())).port,
                            output: 'html',
                        },
                        {
                            extends: 'lighthouse:default',
                            settings: {
                                additionalTraceCategories: TRACE_CATEGORIES.join(','),
                                onlyAudits: LIGHTHOUSE_CATEGORIES.concat('screenshot-thumbnails'),
                                throttlingMethod: 'provided',
                                // throttling:false,
                                emulatedFormFactor: 'none',
                            }
                        }
                    );
                    // console.log("///////////////artifacts:", artifacts)
                    // console.log("audits, artifacts, report :", audits, artifacts, report);
                    if ('pageLoadError-defaultPass' in artifacts.devtoolsLogs) {
                        await sleep(10000);
                        throw Error('Webpage throttling');
                    }

                    // 记录资源数量
                    const { log: { entries } } = chromeHar.harFromMessages(artifacts.devtoolsLogs.defaultPass);
                    //console.log("entries: ", entries)
                    const h2Resources = new Set(entries.filter((entry) => entry.response.httpVersion === 'h2')
                        .map((entry) => entry.request.url));
                    const h3Resources = new Set(entries.filter((entry) => entry.response.httpVersion === 'h3-29')
                        .map((entry) => entry.request.url));
                    const altSvc = new Set(entries.filter((entry) => hasAltSvc(entry))
                        .map((entry) => entry.request.url));
                    
                    //     for(entry in entries)
                    // {
                    //     console.log("--------------------------http version: ", entry.responsse.httpVersion);
                    // }
                    
                    const numH2 = h2Resources.size;
                    const numH3 = h3Resources.size;

                    const difference = new Set([...altSvc].filter((x) => !h3Resources.has(x)));
                    //console.log("numH2, numH3, difference: ", numH2, numH3, difference);

                    const payloadBytes = entries.reduce((acc, entry) => acc + entry.response._transferSize, 0);
                    const payloadMb = (payloadBytes / 1048576).toFixed(2);
                    console.log("--------------------payload Size and number of Resources.------------------");
                    console.log(`Size: ${payloadMb} mb`);

                    console.log("the number of resources:(h2, h3) ", numH2, numH3);
                    console.log("---------------------------------------------------------------------------");
                    

                    if (isH3 && difference.size > 0) {
                        //console.log("difference in runChromeWeb: ",difference);
                        if (urlString === 'https://www.facebook.com/') {
                            domains.push(...entries.filter((entry) => entry.response.httpVersion !== 'h3').map((entry) => entry.request.url));
                        } else {
                            domains.push(...difference);
                        }
                        console.log(`Not enough h3 resources, h2: ${numH2}, h3: ${numH3} `);
                        if (j === RETRIES - 1) {
                            throw Error('Exceeded retries');
                        }
                        continue;
                    }

                    entries.sort((a, b) => (a._requestTime * 1000 + a.time) - (b._requestTime * 1000 + b.time));

                    const start = entries[0]._requestTime * 1000;
                    const end = entries[entries.length - 1]._requestTime * 1000 + entries[entries.length - 1].time;
                    const time = end - start;
                    
                    try {
                        const trace = await wprofx.analyzeTrace(artifacts.traces.defaultPass.traceEvents);
                        console.log("artifacts.traces.defaultPass.traceEvents", artifacts.traces.defaultPass.traceEvents)
                        console.log("trace: ", trace);
                        
                        trace.size = payloadMb;
                        trace.time = time;
                        trace.entries = entries;

                        const plt = trace.loadEventEnd;
                        const fcp = trace.firstContentfulPaint;
                        const wprofxDiff = audits['first-contentful-paint'].numericValue - fcp;
                        timings.plt.push(plt + wprofxDiff);

                        fs.writeFileSync(Path.join(wprofxDir, `wprofx_${i}.json`), JSON.stringify(trace));
                    } catch (error) {
                        console.error(error);
                        throw error;
                    }

                    LIGHTHOUSE_CATEGORIES.forEach((cat) => {
                        timings[cat].push(audits[cat].numericValue);
                    });

                    console.log(`Total: ${entries.length}, h2: ${numH2}, h3: ${numH3}, time: ${audits['speed-index'].numericValue} `);

                    const realImageDir = Path.join(imageDir, `request_${i}`);
                    if (!fs.existsSync(realImageDir)) {
                        fs.mkdirSync(realImageDir, { recursive: true });
                    }
                    audits['screenshot-thumbnails'].details.items.forEach((item, k) => {
                        const base64Data = item.data.replace(/^data:image\/jpeg;base64,/, '');
                        fs.writeFileSync(Path.join(realImageDir, `image_${k}.jpeg`), base64Data, 'base64');
                    });

                } catch (error){
                    throw error;
                } finally {
                    await browser.close();
                }
                const netlogRaw = fs.readFileSync(TMP_NETLOG, { encoding: 'utf-8' });
                let netlog;
                try {
                    netlog = JSON.parse(netlogRaw);
                } catch (error) {
                    // netlog did not flush completely
                    try {
                        netlog = JSON.parse(`${netlogRaw.substring(0, netlogRaw.length - 2)}]}`);
                    } catch (error) {
                        console.log(netlogRaw.substring(netlogRaw.length - 10, netlogRaw.length));
                        throw error;
                    }
                }

                fs.writeFileSync(Path.join(netlogDir, `netlog_${i}.json`), JSON.stringify(netlog));

                //break;

            } catch (error){
                console.log("Retrying......");
                console.error(error);
                if (j === RETRIES - 1) {
                    console.error('Exceeded retries');
                    throw error;
                } 
            }
        }

        fs.writeFileSync(file, JSON.stringify(timings));
    }

    return timings;
};

///--------------------------------------------------
const runBenchmarkWeb = async (urlObj, dirs, isH3, log) => {
    console.log("-------------------runBenchmarkWeb Start.--------------------");
    let timings = { plt: [],};

    LIGHTHOUSE_CATEGORIES.forEach((cat) => {
        timings[cat] = [];
    });

    const realNetlogDir = Path.join(dirs.netlog, `chrome_${isH3 ? 'h3' : 'h2'}_multi`);
    if (!fs.existsSync(realNetlogDir)) {
        fs.mkdirSync(realNetlogDir, { recursive: true });
    }
    const realWprofxDir = Path.join(dirs.wprofx, `chrome_${isH3 ? 'h3' : 'h2'}`);
    if (!fs.existsSync(realWprofxDir)) {
        fs.mkdirSync(realWprofxDir, { recursive: true });
    }
    const realImageDir = Path.join(dirs.images, `chrome_${isH3 ? 'h3' : 'h2'}`);
    if (!fs.existsSync(realImageDir)) {
        fs.mkdirSync(realImageDir, { recursive: true });
    }

    // Read from timings file if exists
    const file = Path.join(dirs.timings, `chrome_${isH3 ? 'h3' : 'h2'}.json`);
    //console.log("timings file path: ", file);
    try {
        timings = JSON.parse(fs.readFileSync(file, 'utf8'));
    } catch (error) {
        //console.log("read timings file error.");
        
    }

    // Run benchmark
    const result = await runChromeWeb(urlObj, timings, file, realNetlogDir, realWprofxDir, realImageDir, isH3, log);
    console.log("result in runBenchmarkWeb: ", result);

    // Get median index of timings
    const siMedianIndex = argsort(timings['speed-index'])[Math.floor(timings['speed-index'].length / 2)];
    const pltMedianIndex = argsort(timings['plt'])[Math.floor(timings['plt'].length / 2)];

    console.log("siMedianIndex, pltMedianIndex: ", siMedianIndex, pltMedianIndex);
    /* // Remove netlogs that are not plt or speed-index median
    fs.readdirSync(realNetlogDir).forEach((f) => {
        const fArr = f.split('.');
        const i = parseInt(fArr[0].split('_')[1], 10);
        if (!(i === siMedianIndex || i === pltMedianIndex)) {
            fs.unlinkSync(Path.join(realNetlogDir, f));
        }
    });

    // Remove traces that are not plt or speed-index median
    fs.readdirSync(realWprofxDir).forEach((f) => {
        const fArr = f.split('.');
        const i = parseInt(fArr[0].split('_')[1], 10);
        if (!(i === siMedianIndex || i === pltMedianIndex)) {
            fs.unlinkSync(Path.join(realWprofxDir, f));
        }
    });

    // Remove image directories that are not speed index median
    fs.readdirSync(realImageDir).forEach((d) => {
        const i = parseInt(d.split('_')[1], 10);
        if (i !== siMedianIndex) {
            deleteFolderRecursive(Path.join(realImageDir, d));
        }
    }); */
};

/////////////////////////////////////////////////////////////////////////
///////10.31 multi-obj web跑通（仅cloudflare），剩single obj-http3部分//////
/////////////////////////////////////////////////////////////////////////


(async () => {
    const parser = new argparse.ArgumentParser();

    parser.add_argument('--dir');
    parser.add_argument('--multi',{ action: argparse.BooleanOptionalAction, help: 'is mutli object (i.e an image resource vs a web-page)', default: false });
    parser.add_argument('--log', { action: argparse.BooleanOptionalAction, help: 'Log netlog', default: false });
    const cliArgs = parser.parse_args();
    const {dir, multi, log} = cliArgs;
    console.log("cliArgs: ",cliArgs);
    
    const clients = CONFIG.clients.filter(client => client.includes("chrome"));
    const sizes = multi ? MULTI_SIZES : SINGLE_SIZES;

    for (const domain of DOMAINS){
        for (const size of sizes){
            //console.log("domian, size : ",domain, size);
            if (!(size in ENDPOINTS[domain])){
                continue;
            }
    
            const urlObj = ENDPOINTS[domain][size];
            console.log("urlObj: ", urlObj);
    
            //创建timings和metrics文件夹
            const dirs = {};
            Object.entries(DIRS).forEach(([key, value]) => {
                dirs[key] = Path.join(value, dir, domain, size);
                //console.log("dirs[key]: ", dirs[key]);
                //console.log("key, value: ",key, value);
                // Only create metrics and timings directories here
    
                if (key === 'timings' && !fs.existsSync(dirs[key])) {
                    fs.mkdirSync(dirs[key], { recursive: true });
                }
                if (key === 'metrics' && !fs.existsSync(dirs[key])) {
                    fs.mkdirSync(dirs[key], { recursive: true });
                }
                
            });
            
            console.log("---------------main start.---------------");
            console.log(`${domain}/${size}`);
    
            for(const client of clients){
                
                const isH3 = client == 'chrome_h3';
                console.log("client: ", client, "isH3: ", isH3);
                //console.log("multi? ", multi);
                if(multi){
                    //console.log(`Chrome: ${isH3 ? 'H3' : 'H2'} - multi object`);
                    await runBenchmarkWeb(urlObj, dirs, isH3, log);
                }


            }
        }
    }
})();



//runBenchmarkWeb inputs: urlObj, dirs, isH3, log;
//urlObj: 
//const isH3 = true;

// console.log(timings);
/*
LIGHTHOUSE_CATEGORIES.forEach((cat) => {
    timings[cat] = [];
    console.log(timings[cat]);
});



const args = chromeArgs(isH3 ? domains : null, log);
puppeteer.launch({
        headless: true,
        defaultViewport: null,
        args,
    }).then(async browser =>{

        
    });


*/
