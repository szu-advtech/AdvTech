//console.log("abcddyyy");
const fs = require('fs');
const url = require('url');
const Path = require('path');
const puppeteer = require('puppeteer');
const PuppeteerHar = require('puppeteer-har');
const puppeteer_core = require('puppeteer-core');
const argparse = require('argparse');
const { setDefaultResultOrder } = require('dns');

const ENDPOINTS = JSON.parse(fs.readFileSync(Path.join(__dirname,'endpoints.json'), 'utf8'));
const CONFIG = JSON.parse(fs.readFileSync(Path.join(__dirname, 'config.json'), 'utf8'));

const RETRIES = 2;
const ITERATIONS = CONFIG.iterations.value;

const DOMAINS = CONFIG.domains.value;
const SINGLE_SIZES = CONFIG.sizes.single;
const MULTI_SIZES = CONFIG.sizes.multi;


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
const CHROME_PROFILE = Path.join(TMP_DIR, 'chrome-profile');
const TMP_NETLOG = Path.join(TMP_DIR, 'chrome.json');
/////////////////////////////////////////
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

////////////////////////////////////////某某排序
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

/////////////////////////////////////////
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

////////////////////////////////////////
const invertMap = (map) => {
    const result = {};

    Object.entries(map).forEach(([key, value]) => {
        result[value] = key;
    });

    return result;
};

////////////////////////////////////////
const getNetlogTime = (netlog) => {
    const logEventTypes = invertMap(netlog.constants.logEventTypes);
    const logEventPhase = invertMap(netlog.constants.logEventPhase);

    let start = 0;
    let end = 0;
    let initRtt = null;

    let firstDataPktTime = null
    let initCwndMss = 0;
    let initCwndBytes = 0;

    for (const event of netlog.events) {
        const eventTime = parseInt(event.time, 10);
        const eventType = logEventTypes[event.type];
        const eventPhase = logEventPhase[event.phase];
        const eventParams = event.params;

        if (eventType === 'TCP_CONNECT') {
            if (eventPhase === 'PHASE_BEGIN') {
                start = eventTime;
            } else {
                initRtt = eventTime - start;
            }
        }
        if (eventType === 'QUIC_SESSION_PACKET_SENT'
            && eventParams['encryption_level'] === 'ENCRYPTION_INITIAL'
            && start === 0) {
            start = eventTime;
        }
        if (eventType === 'QUIC_SESSION_UNAUTHENTICATED_PACKET_HEADER_RECEIVED'
            && eventParams['long_header_type'] === 'INITIAL'
            && initRtt === null) {
            initRtt = eventTime - start;
        }

        if ((eventType === 'HTTP2_SESSION_RECV_HEADERS' || eventType === 'HTTP3_HEADERS_RECEIVED')
            && firstDataPktTime === null) {
            firstDataPktTime = eventTime;
        }

        if (eventType === 'HTTP2_SESSION_RECV_DATA' && eventParams.stream_id === 1) {
            if (firstDataPktTime !== null && eventTime <= firstDataPktTime + initRtt) {
                initCwndBytes += eventParams['size'];
            }
            if (eventParams.fin) {
                end = eventTime;
            }
        }

        if (eventType === 'QUIC_SESSION_STREAM_FRAME_RECEIVED') {
            if (eventParams['stream_id'] === 0 && firstDataPktTime !== null && eventTime <= firstDataPktTime + initRtt) {
                initCwndMss += 1;
                initCwndBytes += eventParams['length'];
            }
            end = eventTime;
        }
    }

    return {
        'time': end - start,
        'init_cwnd_mss': initCwndMss,
        'init_cwnd_bytes': initCwndBytes
    };
};

/////----------------------------------
const runBenchmark = async (urlString, dirs, isH3, log) =>{
    let timings = [];
    let metrics = [];

    console.log("isH3 in runBenchmark: ", isH3);
    //为h2 h3创建日志文件夹
    const realNetlogDir = Path.join(dirs.netlog, `chrome_${isH3 ? 'h3' : 'h2'}_single`);
    if (!fs.existsSync(realNetlogDir)) {
        fs.mkdirSync(realNetlogDir, { recursive: true });
    }

    //如果存在，读取timings和metrics文件
    const timings_file = Path.join(dirs.timings, `chrome_${isH3 ? 'h3' : 'h2'}.json`);
    try {
        timings = JSON.parse(fs.readFileSync(timings_file, 'utf8'));
    } catch (error) {
        //
        console.log(`timing file is not exist.(chrome_${isH3 ? 'h3' : 'h2'}.json)`);
    }
    const metrics_file = Path.join(dirs.metrics, `chrome_${isH3 ? 'h3' : 'h2'}.json`);
    try {
        metrics = JSON.parse(fs.readFileSync(metrics_file, 'utf8'));
    } catch (error) {
        //
        console.log(`metrics file is not exist.(chrome_${isH3 ? 'h3' : 'h2'}.json)`);
    }

    //console.log("timing.length, ITERATIONS: ", timings.length, ITERATIONS);
    
    if (timings.length >= ITERATIONS) {
        return;
    }

    // Run benchmark
    const result = await runChrome(urlString, realNetlogDir, isH3, timings.length, log);


    // Concat result times to existing data
    timings.push(...result.timings);
    metrics.push(...result.metrics);

    // Save data
    fs.writeFileSync(timings_file, JSON.stringify(timings));
    fs.writeFileSync(metrics_file, JSON.stringify(metrics));

    // Get median index of timings
    const medianIndex = argsort(timings)[Math.floor(timings.length / 2)];

    // Remove netlogs that are not median
    fs.readdirSync(realNetlogDir).forEach((f) => {
        const fArr = f.split('.');
        const i = parseInt(fArr[0].split('_')[1], 10);
        if (i !== medianIndex) {
            fs.unlinkSync(Path.join(realNetlogDir, f));
        }
    });

}
/////----------------------------------
const runChrome = async (urlString, netlogDir, isH3, n, log) => {
    const metrics = [];
    const timings = [];

    console.log("isH3 in runChrome: ", isH3);
    console.log(`${urlString}`);

    /////here is different with the origin code.
    /*
    if (urlString_local.includes('speedtest-100KB')) {
        gotoUrl = `file://${Path.join(__dirname, 'html', '100kb.html')}`;
    } else if (urlString_local.includes('speedtest-1MB')) {
        gotoUrl = `file://${Path.join(__dirname, 'html', '1mb.html')}`;
    } else if (urlString_local.includes('speedtest-5MB')) {
        gotoUrl = `file://${Path.join(__dirname, 'html', '5mb.html')}`;
    } else {
        gotoUrl = urlString_local;
    }
    */
    urlString_local = "https://127.0.0.1:443/test/pulpit.jpg";
    urlString_local_ols = "https://127.0.0.1:443/cgi-bin/helloworld";
    const urlSting_xquic_server = "https://127.0.0.1:8443";
    const urlQuic_google = "https://quic.rocks:4433";
    const urlQuic_cloudflare = "https://cloudflare-quic.com/";

    let gotoUrl = urlString_local_ols;
    //console.log("gotoUrl defined by urlString.", gotoUrl);

    for(let i = n; i < ITERATIONS; i += 1){
        if(netlogDir.includes('LTE')){
            await sleep(10000);
        }

        console.log(`/////-----//////Iteration: ${i}`);

        for (let j = 0; j < RETRIES; j += 1){
            try{
                deleteFolderRecursive(CHROME_PROFILE);
                const args = chromeArgs(isH3 ? [gotoUrl] : null, log);
                args.push('--no-sandbox');
                args.push('--allow-insecure-localhost');//自建证书必加该行，本地服务器使用
                
                // console.log("------args:", args);
        
                const browser = await puppeteer.launch({
                    headless: true,
                    defaultViewport: null,
                    args,
                });
        
                try{
                    console.log("-----browser.newPage-----");
                    const page = await browser.newPage();
                    
                    const har = await new PuppeteerHar(page);
                    await har.start();
                    console.log("gotoUrl in page.goto: ",gotoUrl);
                    await page.goto(gotoUrl,{
                        timeout: 12000,
                        });
                    //console.log("here?");
                    const harResult = await har.stop();
                    const { entries } = harResult.log;
                    await page.close();
                    //console.log("entries:", entries);
                    const result = entries.filter((entry) => entry.request.url != urlString_local);
                    //console.log("///****///result: ", result);
                    
                    if(result.length != 1){
                        // console.error("Invalid HAR", result);
                        throw Error;
                    }

                    const entry = result[0];
                    console.log("------entry.request.httpVersion:  ", entry.request.httpVersion);
                    console.log("------entry.response.httpVersion:  ", entry.response.httpVersion);
                    
                    if(entry.response.status !== 200){
                        console.error('Unsuccessful request');
                        throw Error;
                    }


                    const harTime = entry.time - entry.timings.blocked - entry.timings._queued - entry.timings.dns;
                    //console.log("-----harTime: ", harTime);
                    console.log(entry.response.httpVersion, harTime);
        
                    if(isH3 && entry.response.httpVersion !== 'h3-29'){
                        console.log("the protocol is: ", entry.response.httpVersion);
                        throw Error('incorrect protocol');
                    }
        
                    if(!isH3 && entry.response.httpVersion !== 'h2'){
                        console.log("the protocol is: ", entry.response.httpVersion);
                        throw Error('incorrect protocl');
                    }
        
                    await browser.close();
        
                    if(!log){
                        timings.push(harTime);
                        break;
                    }
                    const netlogRaw = fs.readFileSync(TMP_NETLOG, { encoding: 'utf-8' });
                    let netlog;
                    try {
                        // console.log("-----try correct.");
                        netlog = JSON.parse(netlogRaw);
                    } catch (error) {
                        // netlog did not flush completely
                        console.log("netlog did not flush completely");
                        netlog = JSON.parse(`${netlogRaw.substring(0, netlogRaw.length - 2)}]}`);
                    }
            
                    const res = getNetlogTime(netlog);
                    const time = res.time;
                    console.log("res------: ", res);
                    // console.log(res); 
                    console.log('netlog time:', time);
                    metrics.push(res);
                    timings.push(time);
                    fs.writeFileSync(Path.join(netlogDir, `netlog_${i}.json`), JSON.stringify(netlog));
                    //console.log("---*****----netlogDir: ", netlogDir);
        
                    break;
        
                } catch (error){
                    await browser.close();
                    console.error(error);
                    if(j === RETRIES - 1){
                        console.error('Exceeded retries');
                        throw error;
                    }
        
                }
            } catch(error){
                console.error(error);
                if(j === RETRIES - 1){
                    console.error('Exceeded retries');
                    throw error;
                }
            }
        }
    }
    
    return { timings, metrics};
}


(async () => {
    /* 命令行设置：dir即为输入的结果存放文件夹*/
    console.log("start?");
    const parser = new argparse.ArgumentParser();
    parser.add_argument('--dir');
    parser.add_argument('--multi', { action: argparse.BooleanOptionalAction, help: 'is mutli object (i.e an image resource vs a web-page)', default: false });
    parser.add_argument('--log', { action: argparse.BooleanOptionalAction, help: 'Log netlog', default: false });

    const cliArgs = parser.parse_args();
    const{dir,multi,log} = cliArgs;//cliArgs: Namespace(dir='test_1', multi=true, log=true)
    console.log("cliArgs:", cliArgs);

    const sizes = SINGLE_SIZES;
    const clients = CONFIG.clients.filter(client => client.includes("chrome"));
    // console.log("CONFIG.clients: ",CONFIG.clients);

    console.log("sizes: ", sizes);

    
    for(const domain of DOMAINS){
        // console.log("domine: ",domain);
        for(const size of sizes){
            if (!(size in ENDPOINTS[domain])) {
                continue;
            }

            const urlObj = ENDPOINTS[domain][size];
            console.log("BEGINNING: ");
            const dirs = {};
            //创建timings和metrics文件夹
            Object.entries(DIRS).forEach(([key,value]) => {
                dirs[key] = Path.join(value, dir, domain, size);
                
                //console.log("dirs[key]: ", dirs[key]);
                //console.log("key, value: ",key, value);
                
                if(key === 'timings' && !fs.existsSync(dirs[key])){
                    fs.mkdirSync(dirs[key], { recursive: true });
                }
                if (key === 'metrics' && !fs.existsSync(dirs[key])) {
                    fs.mkdirSync(dirs[key], { recursive: true });
                }
                
            })

            console.log(`${domain}/${size}`);
            
            for (const client of clients) {
                console.log("clients in for: ", client);
                const isH3 = client == 'chrome_h3'
                //const isH3 = false;
                //console.log("isH3 in main: ", isH3);
                
                console.log(`Chrome: ${isH3 ? 'H3' : 'H2'} - single object`);
                await runBenchmark(urlObj, dirs, isH3, log);
                
            }
        }
        
    }
})();


/* runBenchmark(urlObj, dirs, isH3, log)
inputs:(runBenchmarkWeb is the same)
rulObj = ENDPOINTS[domain][size]
       = "https://storage.googleapis.com/gweb-uniblog-publish-prod/images/logo_android_auto_color_2x_web_512dp.max-600x600.png"
dirs = {[key,value]*6} eg:timings: '/home/sky/quic-benchmarks/data/timings/test_1/brown/100KB'
isH3 = client = 'chrome_h3', CONFIG.clients:  [ 'curl_h2', 'chrome_h2', 'chrome_h3', 'proxygen_h3', 'ngtcp2_h3' ]
log = --log default = false;
*/
////////runChrome test///////////
/*runChrome(urlString, realNetlogDir, isH3, timings.length, log)
urlString = urlObj, like above;
realNetlogDir = .../data/netlog/test_1/brown/100KB/chrome_h3_single
isH3 = true(if is 'chrome_h3')
timgings.length = 0(chrome_h3.json is not exist in file' timings)
log: like above, is a bool.
*/



//100KB
//urlObj: https://127.0.0.1:443/test
/*
urlString_google = "https://storage.googleapis.com/gweb-uniblog-publish-prod/images/logo_android_auto_color_2x_web_512dp.max-600x600.png";
urlString_local = "https://127.0.0.1/test";
urlString_baidu = "https://www.baidu.com";
urlString_facebook = "https://scontent.xx.fbcdn.net/speedtest-100KB";
const isH3 = false; //clients = "chrome_h2"
const realNetlogDir = Path.join(dirs.netlog,`chrome_${isH3 ? 'h3' : 'h2'}_single`);
const netlogDir = realNetlogDir;
*/
//let gotoUrl = urlString_local;


//console.log("args: ",args1);
/*const args = [
        '--no-sandbox',
        '--headless',
        //'--disable-setuid-sandbox',//ewai
        //'--unhandled-rejections=strict',
        '--disable-gpu',
        '--disable-dev-shm-usage',
        // '--window-size=1920,1080',
        `--user-data-dir=${CHROME_PROFILE}`,
        '--disk-cache-dir=/dev/null',
        '--disk-cache-size=1',
        '--aggressive-cache-discard',
        `--log-net-log=${TMP_NETLOG}`,
        '--disable-quic',
        //'--enable-quic',
        //'--quic-version=h3-29',
]
*/

    
    /*puppeteer.launch({
        headless: true,
        defaultViewport: null,
        args,
    }).then(async browser =>{
}*/
        
        

        //console.log(entries.request);
        
        
        

        
//////////////////////////////////////////////////////////////////////////////////////////
/////////10.28 至此，runbenchmark函数已经整体跑通：h2 client and h2 server with apache2///////
////////其后，尝试在原chrome.js中更换链接跑一次，然后开始研究runbenchmarkWeb函数//////////////////
//////////////////////////////////////////////////////////////////////////////////////////



/*const page = browser.newPage();
const har = new PuppeteerHar(page);
har.start()
page.goto(gotoUrl,{
    timeout: 12000,
});*/
/* 一些输出信息 
console.log("dirs: ", dirs);
console.log(dirs.netlog);

console.log("realNetlogDir: ", realNetlogDir);
if (!fs.existsSync(realNetlogDir)) {
    fs.mkdirSync(realNetlogDir, { recursive: true });
    console.log("dir Running as root without --no-sandbox is not supported. made.");
}
const timings_file = Path.join(dirs.timings, `chrome_${isH3 ? 'h3' : 'h2'}.json`);
try {
    timings = JSON.parse(fs.readFileSync(timings_file, 'utf8'));
} catch (error) {
    console.log("timing JSON error.")
    //
}
console.log(timings_file);
console.log("timing.length: ", timings.length);

*/
/*
if(netlogDir.includes('LTE')){
    console.log("yes.1");
}
else{
    console.log("no.1");
}
*/

////////runChrome end////////////////

//console.log(CONFIG);
/*const sizes = SINGLE_SIZES;
for(const domain of DOMAINS){
    console.log("domine: ",domain);
    for(const size of sizes){

        if (!(size in ENDPOINTS[domain])) {
            continue;
        }

        console.log("sizes in domain: ",size);
        console.log("endpoints.domain.size: ",ENDPOINTS[domain][size])
    }
    
}
//console.log(ENDPOINTS[DOMAINS]);

const path1 = Path.join(__dirname, 'endpoints.json');
console.log(path1);
*/