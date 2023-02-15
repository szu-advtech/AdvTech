const util = require('util');
const child_process = require('child_process');
const exec = util.promisify(child_process.exec);
$("#btnId").click(async function(cb) {
  exec('sh printtest.sh aist_entrance_hall_1', function (err, stdout, stderr) {
    console.log(stdout);
    console.log(stderr);
    cb(err);
  });
window.location.href="result.html";
});
