export async function api<T>(path:string,method:'GET'|'POST'='GET',body?:unknown):Promise<T>{
 if(!window.desktop)throw new Error('Open this application through Electron to connect to the local research service.');
 return window.desktop.request<T>(path,method,body);
}
