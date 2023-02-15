import axios from 'axios'
let base = 'http://127.0.0.1:5000/';


export const postRequest = (url, params,data) => {
  return axios({
    method: 'post',
    url: `${base}${url}`,
    params: params,
    data:data,
    headers: {
      'Authorization': localStorage.getItem('token'),
      'Content-Type': 'application/json'
    }
  });
}

export const uploadFileRequest = (url, params) => {
  return axios({
    method: 'post',
    data:param,
    url: `${base}${url}`,
    transformRequest: [function (data) {
      let ret = ''
      for (let it in data) {
        ret += encodeURIComponent(it) + '=' + encodeURIComponent(data[it]) + '&'
      }
      return ret
    }],
    headers: {
      'Authorization': localStorage.getItem('token'),
      'Content-Type': 'application/x-www-form-urlencoded'
    }
  });
}
export const putRequest = (url, params) => {
  return axios({
    method: 'put',
    url: `${base}${url}`,
    data: params,
    headers: {
      'Authorization': localStorage.getItem('token'),
      'Content-Type': 'application/json'
    }
  });
}
export const deleteRequest = (url,param,data) => {
  return axios({
    method: 'delete',
    param:param,
    data:data,
    url: `${base}${url}`,
    transformRequest: [function (data) {
      let ret = ''
      for (let it in data) {
        ret += encodeURIComponent(it) + '=' + encodeURIComponent(data[it]) + '&'
      }
      return ret
    }],
    headers: {
      'Authorization': localStorage.getItem('token'),
      'Content-Type': 'application/x-www-form-urlencoded'
    }
  });
}
export const getRequest = (url,params) => {
  return axios({
    method: 'get',
    params:params,
    transformRequest: [function (data) {
      let ret = ''
      for (let it in data) {
        ret += encodeURIComponent(it) + '=' + encodeURIComponent(data[it]) + '&'
      }
      return ret
    }],
    headers: {
      'Authorization': localStorage.getItem('token'),
      'Content-Type': 'application/x-www-form-urlencoded'
    },
    url: `${base}${url}`
  });
}
