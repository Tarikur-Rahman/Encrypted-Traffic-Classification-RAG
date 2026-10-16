import {_electron as electron} from 'playwright';
import {createRequire} from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
const require=createRequire(import.meta.url);
const acceptance=JSON.parse(fs.readFileSync('runtime/final-v5-external-acceptance.json','utf8'));
const chosen=acceptance.inputs.find(row=>row.label==='CIC CSV');
const env={...process.env,PG_PROJECT_ROOT:path.resolve('..'),PG_DEV:'0'};delete env.ELECTRON_RUN_AS_NODE;
const app=await electron.launch({executablePath:require('electron'),args:['.'],env});
const out=path.resolve('runtime/redesign_audit/screenshots');fs.mkdirSync(out,{recursive:true});
try{
 const page=await app.firstWindow();await page.getByText('Local service connected').waitFor({timeout:60000});
 await page.evaluate(async({id,flow})=>{
  const status=await window.desktop.request(`/api/analysis/${id}/status`);
  localStorage.setItem('provenanceguard-session-v1',JSON.stringify({state:{analysisId:id,selectedFlowId:flow,input:status.input,autoNavigationActive:false},version:0}));
 },{id:chosen.analysis_id,flow:chosen.flows[0].flow_id});
 await page.reload();await page.getByText('Local service connected').waitFor({timeout:60000});
 await page.evaluate(()=>{location.hash='/phase1'});
 await page.getByText('Traffic, classified.',{exact:true}).waitFor({timeout:60000});
 const report=[];
 for(const [route,name] of [['/','overview'],['/new','new-analysis'],['/progress','progress'],['/phase1','classification'],['/flows','flows'],['/flow','flow-detail'],['/evidence','evidence'],['/provenance','provenance'],['/reasoning','reasoning'],['/decision','decision']]){
  await page.evaluate(route=>{location.hash=route},route);await page.waitForTimeout(1200);
  if(route==='/phase1')await page.getByText('Running frozen model comparison…').waitFor({state:'hidden',timeout:60000});
  for(const [width,height] of [[1280,720],[1440,900],[1920,1080]]){
   await app.evaluate(({BrowserWindow},{width,height})=>BrowserWindow.getAllWindows()[0].setContentSize(width,height),{width,height});
   await page.waitForTimeout(150);
   const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
   report.push({name,width,height,overflow});
   await page.screenshot({path:path.join(out,`${name}-${width}.png`),fullPage:true});
  }
 }
 fs.writeFileSync('runtime/redesign_audit/visual-report.json',JSON.stringify(report,null,2));
 if(report.some(row=>row.overflow))throw new Error('Horizontal overflow detected');
}finally{await app.close()}
