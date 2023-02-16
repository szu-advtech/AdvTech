<template>
  <div>
  <app-header/>
    <el-container class="main-container" >
      <el-aside >
  <el-card  >
    <div slot="header">
      <div class="operation">
        <el-button  type="info" size="medium" @click="submit">Submit</el-button>
      </div>
    </div>
    <div v-for=" item in canvas.children[0].children" :key="item.id">
    <div>
     <div class="group-node"> 
      - {{item.id}}
    </div>
    <div v-if="item.type!='repeat group'">
    <div  v-for="item2 in item.children" :key="item2.id" class="icon-node"  >
      <img    :src="item2.icon"  />
    </div>
  </div>
  <div v-else style="padding-left: 30px;">
    <div  v-for="item2 in item.children" :key="item2.id" >
      <div class="group-node"> 
      - {{item2.id}}
      </div>
      <div  v-for="item3 in item2.children" :key="item3.id" class="icon-node"  >
      <img    :src="item3.icon"  />
    </div>
    </div>
  </div>
  </div>

  </div>
  </el-card>
      </el-aside>
  <el-main>
    <div class="exam-container">
         <el-card  v-for=" item in result" :key="result.indexOf(item)"  class="box-card">
            <img v-for="element in item"  :key="item.indexOf(element)" :src="element.icon" :style=" 'position:absolute;width:'+ element.size.width+'px;height:'+element.size.height+'px;top:'+ element.location.y+'px;left:'+element.location.x+'px'">
          </el-card>
     </div>
  </el-main>
    </el-container>
  
  </div>  
  </template>
  
  <script>
  import Header from '@/components/Header'
  import { postRequest } from '../utils/api'


  export default {
      data() {
          return{
            canvas:
        {
          id:'canvas',
          type:'canvas',
          order:'unimportant',
          width: 375 ,
          height: 670,
          children:[
            {
              id:"page",
              type:'page',
              order:'unimportant',
              children:[
                {
                  id:'Alternative Group',
                  type:"group",
                  order:'unimportant',
                  children:[
                  {
                    id:'milkshake1',
                    type: 'leaf',
                    width: 100 ,
                    height: 100,
                    importance:'high',
                    icon:'https://scout-1304413189.cos.ap-guangzhou.myqcloud.com/icon/milkshake1.svg'
                  },
                  {
                    id:'milkshake2',
                    type: 'leaf',
                    width: 100 ,
                    height: 100,
                    importance:'high',
                    icon:'https://scout-1304413189.cos.ap-guangzhou.myqcloud.com/icon/milkshake2.svg'
                  },
                  {
                    id:'milkshake3',
                    type: 'leaf',
                    width: 100 ,
                    height: 100,
                    importance:'high',
                    icon:'https://scout-1304413189.cos.ap-guangzhou.myqcloud.com/icon/milkshake3.svg'
                  }
                  ]
                },
                {
                  id:'Order Group',
                  type:"group",
                  order:'important',
                  children:[
                  {
                    id:'smoothie',
                    type: 'leaf',
                    width: 220 ,
                    height: 30,
                    importance:'high',
                    icon:'https://scout-1304413189.cos.ap-guangzhou.myqcloud.com/icon/smoothie.svg'
                  },
                  {
                    id:'line',
                    type: 'leaf',
                    width: 160 ,
                    height: 20,
                    importance:'normal',
                    icon:'https://scout-1304413189.cos.ap-guangzhou.myqcloud.com/icon/line.svg'
                  },
                  {
                    id:'zoey',
                    type: 'leaf',
                    width: 100 ,
                    height: 30,
                    importance:'low',
                    icon:'https://scout-1304413189.cos.ap-guangzhou.myqcloud.com/icon/zoey.svg'
                  }
                  ]
                },
                {
                  id:'Reapeat Group',
                  type:"repeat group",
                  order:'unimportant',
                  children:[
                  {
                  id:'Subgroup 1',
                  type:"group",
                  order:'important',
                  children:[
                  {
                    id:'time',
                    type: 'leaf',
                    width: 45 ,
                    height: 45,
                    importance:'normal',
                    icon:'https://scout-1304413189.cos.ap-guangzhou.myqcloud.com/icon/time.svg'
                  },
                  {
                    id:'time-tag',
                    type: 'leaf',
                    width: 120 ,
                    height: 45,
                    importance:'normal',
                    icon:'https://scout-1304413189.cos.ap-guangzhou.myqcloud.com/icon/time-tag.svg'
                  }
                  ]
                },
                {
                  id:'Subgroup 2',
                  type:"group",
                  order:'important',
                  children:[
                  {
                    id:'fire',
                    type: 'leaf',
                    width: 45 ,
                    height: 45,
                    importance:'normal',
                    icon:'https://scout-1304413189.cos.ap-guangzhou.myqcloud.com/icon/fire.svg'
                  },
                  {
                    id:'fire-tag',
                    type: 'leaf',
                    width: 120 ,
                    height: 45,
                    importance:'normal',
                    icon:'https://scout-1304413189.cos.ap-guangzhou.myqcloud.com/icon/fire-tag.svg'
                  }
                  ]
                }
                  ]
                },
                {
                  id:'Group',
                  type:"group",
                  order:'unimportant',
                  children:[
                  {
                    id:'button',
                    type: 'leaf',
                    width: 118 ,
                    height: 40,
                    importance:'low',
                    icon:'https://scout-1304413189.cos.ap-guangzhou.myqcloud.com/icon/button.svg'
                  }
                  ]
                }
              ]
            }
          ]
        },
              result:[]
          }
      },
      components:{
          'app-header':Header,
      },
      methods:{
     
    submit(){
      let canvas = JSON.parse(JSON.stringify(this.canvas))
        canvas.children[0].children[0].children.pop()
        canvas.children[0].children[0].children.pop()
        canvas.children[0].children[0].children[0].id = "milkshake"
        var _this = this;
        postRequest('solve',null,
          {
            elements: [canvas]
          }
        ).then(resp=>{
          _this.result = []
          let res_list = resp.data.solutions
          for(var i=0;i<res_list.length;i++){
            let temp_element_list = []
            for(var j in res_list[i].elements)
            {
              if(!res_list[i].elements[j].hasOwnProperty('children')){
                temp_element_list.push(res_list[i].elements[j])
              }
              if(res_list[i].elements[j].id == 'milkshake'){
                var alt = Math.round(Math.random() * 2 ) + 1;
                res_list[i].elements[j].icon = 'https://scout-1304413189.cos.ap-guangzhou.myqcloud.com/icon/milkshake' + alt + '.svg'
              }
            }
            _this.result.push(temp_element_list)
          }
         // console.log(resp.data.solutions)
          console.log(_this.result) 
          //_this.$forceUpdate()

        },resp=>{
            console.log(resp);
        })

      },
      paintres(){
        for(var i=0;i<this.result.length;++i)
          {
            for(var j=0;j<this.result[i].length;j++){
              
              var x = this.result[i][j].location.x;
              var y = this.result[i][j].location.y;
              //console.log(x)
              //console.log(y)
              var img
              if(this.result[i][j].id=='milkshake1'){
                  var alt = Math.round(Math.random() * 1 ) + 1;
                  img = document.getElementById('milkshake'+alt);
              }

              else img = document.getElementById(this.result[i][j].id);
              //console.log(img)
              var canvas = document.getElementById('canvas'+i)
              //console.log(canvas)
              var context = canvas.getContext('2d')
              context.drawImage(img, x,y);
            }
          }
      }
      },
      created(){
       
      }
  }
  </script>
  
  <style>
    .main-container{
        position:fixed; 
        width: 100%;
        height: 100% ; 
        top: 0; 
        left:0;
    }
    .el-main{
        padding-top: 60px;
        position:relative; 
        width: 70%;
        height: 100%; 
        bottom: 0; 
        right:0;
    }
    .el-aside{
        padding-top:60px ;
        position:relative; 
        width: 30%;
        height: 100% ; 
        bottom: 0; 
        left:0;
    }
    .editor{
        height: 300px;
        position: relative;
        bottom: 0;
        width: 100%;
  
    }
  
    .question-button{
        width:40px;margin-right:5px; margin-bottom:5px;
    }
    .statistic, .operation, .current{
        display: flex;
        justify-content: space-around;
        font-size:16px;
        margin-bottom: 10px;
    }
    .exam-info{
        display: flex;
        justify-content: space-around;
        font-size:18px;
        margin-bottom: 10px;
    }
    .question-type{
        padding-top:18px;
        font-size:16px;
        color:grey;
    }
    
   .question-choice,  .content-bar{
        margin-bottom: 10px;
        font-size:18px;
    }
  
  .answer-area, .explanation, .feedback{
      padding: 10px 0
  }
  
  .feedback-item{
      margin: 5px 0 ;
      font-size:20px;
  }
  
  .submit-button{
      margin:10px 0
  }
  .icon-node {
  margin-top: 10px;
  flex: 1;
  display: flex;
  align-items:center;
  justify-content: space-between;
  font-size: 14px;
  padding-right: 8px;
  margin-bottom:10px;
  padding-left:30px;
}

.group-node {
  margin-top: 10px;
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-right: 8px;
  margin-bottom:10px;
}

.box-card {
    margin: 5px;
    width:375px;
    height: 670px;
    position:relative;
  }
  .exam-container{
      display: flex;
      flex-wrap:wrap;
  }

  </style>

