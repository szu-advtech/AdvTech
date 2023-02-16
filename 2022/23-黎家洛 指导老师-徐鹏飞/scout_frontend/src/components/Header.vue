<template>
<div class="outter">
    <el-container class="home_container">
    <el-header>
        <router-link to="/" style="text-decoration:none;color:black">
            <div class="home_title">SCOUT</div>
        </router-link>
      <div class="hearderComponent">
          </div>
    </el-header>
    </el-container>
</div>
       
</template>
<script>


import {getRequest} from '../utils/api';
export default {
    data() {
        return {
           
        }
    },
    created:function(){
   
    },
    methods:{
       
    }
}
</script>
<style >

.home_container {
    position: fixed;
    top: 0px;
    left: 0px;
    width: 100%;
    z-index: 999;
  }

.el-header {
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background-color: white;
    border-bottom-style:ridge;
   
  }
  .home_title {
    
    font-size: 20px;
    display: inline;

  }

  .home_userinfo {
    color: #fff;
    cursor: pointer;
  }

  .home_userinfoContainer {
    display: inline;

   
  }
  .home_searchContainer {
    display: inline;
    margin-right: 30px;
  }

.hearderComponent {
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
 
.outter{
  margin-bottom: 60px;
}

</style>
