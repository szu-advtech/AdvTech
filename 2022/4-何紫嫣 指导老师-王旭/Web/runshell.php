<?php

function upload() {
	if (!empty($_FILES['file'])) {


	$target = "upload/". $_FILES["file"]["name"];

	if (move_uploaded_file($_FILES['file']['tmp_name'], $target))
		return ['uploaded' => $target];
	else return ['error'=>'save error'];
	} 
	else return ['error'=>'No files found for upload.'];

}

// 设置允许其他域名访问
header('Access-Control-Allow-Origin:*');  
// 设置允许的响应类型 
header('Access-Control-Allow-Methods:POST');  
// 设置允许的响应头 
header('Access-Control-Allow-Headers:x-requested-with,content-type'); 
// header("Access-Control-Allow-Headers: Origin, X-Requested-With, Content-Type, Accept");
header('Content-Type: application/json'); // set json response headers
$outData = upload(); // a function to upload the bootstrap-fileinput files

set_time_limit(0);
ob_end_clean();
header("Connection: close");
header("HTTP/1.1 200 OK");
ob_start();
echo json_encode($outData); // return json data
$size = ob_get_length();
header("Content-Length: $size");
ob_end_flush();
flush();
if (function_exists("fastcgi_finish_request")) { // yii或yaf默认不会立即输出，加上此句即可（前提是用的fpm）
    fastcgi_finish_request(); // 响应完成, 立即返回到前端,关闭连接
}
sleep(2);
ignore_user_abort(true);// 在关闭连接后，继续运行php脚本
set_time_limit(0);

$shell = "sh vslam.sh 2>&1";
exec($shell, $result, $status);
$shell = "<font color='red'>$shell</font>";
echo "<pre>";
if( $status ){
echo "shell命令{$shell}执行失败";
var_dump($result);
} else {
echo "shell命令{$shell}成功执行, 结果如下<hr>";
print_r( $result );
}
echo "</pre>";
// echo "<script>window.location.href=‘result.html’</script>";
// header("Refresh:3;url=result.html");
?>