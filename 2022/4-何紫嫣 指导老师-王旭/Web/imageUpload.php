<?php
// 设置允许其他域名访问
header('Access-Control-Allow-Origin:*');  
// 设置允许的响应类型 
header('Access-Control-Allow-Methods:POST');  
// 设置允许的响应头 
header('Access-Control-Allow-Headers:x-requested-with,content-type'); 
// header("Access-Control-Allow-Headers: Origin, X-Requested-With, Content-Type, Accept");
header('Content-Type: application/json'); // set json response headers
// echo "Hello world!<br>";
// print_r($_FILES['file']);
// echo '<br/>file error='.$_FILES['file']['error'] ;
$outData = upload(); // a function to upload the bootstrap-fileinput files
exec("sh printtest.sh aist_entrance_hall_1", $output,$return);
echo json_encode($outData); // return json data
exit(); // terminate

function upload() {
	if (!empty($_FILES['file'])) {


	$target = "upload/". $_FILES["file"]["name"];

	if (move_uploaded_file($_FILES['file']['tmp_name'], $target))
		return ['uploaded' => $target];
	else return ['error'=>'save error'];
	} 
	else return ['error'=>'No files found for upload.'];

}

?>