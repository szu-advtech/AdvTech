
const path = require('path');
const gulp = require('gulp');
const exec = require('child_process').exec;

const fs = require("fs");
const connect = require('gulp-connect');
const livereload = require('gulp-livereload');
// var watch = require('gulp-watch');
const {watch} = gulp;

// For development, it is now possible to use 'gulp webserver'
// from the command line to start the server (default port is 8080)
gulp.task('webserver', gulp.series(async function() {
	server = connect.server({
		port: 1245,
		// https: false,
		host: '0.0.0.0',
		livereload:true,
	});
}));

// gulp.task("pack", async function(){
// 	exec('rollup -c', function (err, stdout, stderr) {
// 		console.log(stdout);
// 		console.log(stderr);
// 	});
// });

gulp.task('html', async function(){
	gulp.src('./tsdf.html').pipe(livereload()).pipe(connect.reload());
});

gulp.task('converter', async function(cb) {
  exec('sh converter.sh', function (err, stdout, stderr) {
    console.log(stdout);
    console.log(stderr);
    cb(err);
  });
})

gulp.task('watch', gulp.parallel("html", "webserver", async function() {
	livereload.listen();
	// let watchlist = [
	// 	// './upload/getresult.js',
	// 	'/home/code/openvslam-comments/tsdf_new.ply',
	// ];

	watch('./upload/tsdf_new.ply', gulp.series("converter"));	
	watch('./upload/getresult.js', gulp.series("html"));	
	// let watchlist = [
	// 	'./result.html',
	// ];

	// watch(watchlist, gulp.series("vslam"));

}));


