import {contextBridge,ipcRenderer} from 'electron';
contextBridge.exposeInMainWorld('desktop',{
 selectTrafficFile:(kind?:'csv'|'pcap')=>ipcRenderer.invoke('select-file',kind),
 selectExportLocation:(format:string)=>ipcRenderer.invoke('select-export',format),
 exportAnalysis:(id:string,format:string)=>ipcRenderer.invoke('export-analysis',id,format),
 getAppVersion:()=>ipcRenderer.invoke('app-version'),
 openExternalDocument:(url:string)=>ipcRenderer.invoke('open-document',url),
 request:(path:string,method='GET',body?:unknown)=>ipcRenderer.invoke('api-request',path,method,body),
});
