/*
   Copyright 2012 IBM

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

window.Test = Backbone.Model.extend({

    defaults: {
        macAddress:' ',
        ipv6:' ',
        switchport:' ',
        time:''
    },

    initialize:function () {}

});

window.TestCollection = Backbone.Collection.extend({

    model:Test,
    
    // instead of the collection loading its children, the switch will load them
    initialize:function () {
    //fetch:function(){
    	/*var self=this;
    
    	$.ajax({
            //提交数据的类型 POST GET
            type:"POST",
            //提交的网址
            url:"/savi/config",
            //提交的数据
            data:{"type":"type"},
            //返回数据的格式
            datatype: "json",//"xml", "html", "script", "json", "jsonp", "text".
            //成功返回之后调用的函数             
            success:function(data){
        		var old_ids=self.pluck('id');
            	_.each(data,function(test)){
            		old_ids=_.without(old_ids,test['id']);
            		self.add({macAddress:test['mac'],ipv6:test['ip'],switchport:test['dpid']+'-'+test['port'],time:test['binding-time']});
            	});
    	
    			_.each(old_ids,function(test){
    				self.remove({id:test});
    			});
            	self.trigger('add');
            	
            }         
         });
    	*/
    }
    
});