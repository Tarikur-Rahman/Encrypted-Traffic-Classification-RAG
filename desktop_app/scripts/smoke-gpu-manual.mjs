import {_electron as electron} from 'playwright';
import {createRequire} from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
const require=createRequire(import.meta.url);
const accepted=JSON.parse(fs.readFileSync('runtime/final-v5-external-acceptance.json','utf8')).inputs.find(row=>row.label==='CIC CSV');
const env={...process.env,PG_DEV:'0',PG_PROJECT_ROOT:path.resolve('..')};delete env.ELECTRON_RUN_AS_NODE;
const app=await electron.launch({executablePath:require('electron'),args:['.'],env,timeout:60000});
const report={passed:false,viewports:[],manualStarts:[],errors:[]};
const directory='runtime/gpu_manual_audit';fs.mkdirSync(`${directory}/screenshots`,{recursive:true});
try{
 const page=await app.firstWindow();page.on('pageerror',e=>report.errors.push(String(e)));
 await page.getByText('Local service connected').waitFor({timeout:60000});
 const flows=await page.evaluate(async id=>(await window.desktop.request(`/api/analysis/${id}/flows`)).rows,accepted.analysis_id);
 async function select(flow){
  await page.evaluate(async({id,flow})=>{
   const status=await window.desktop.request(`/api/analysis/${id}/status`);
   localStorage.setItem('provenanceguard-session-v1',JSON.stringify({state:{analysisId:id,selectedFlowId:flow,input:status.input},version:0}));
  },{id:accepted.analysis_id,flow});
  await page.reload();await page.getByText('Local service connected').waitFor({timeout:60000});
 }
 await select(flows[1].id);
 await page.evaluate(()=>{location.hash='/phase1'});
 await page.getByText('Traffic, classified.',{exact:true}).waitFor({timeout:60000});
 await page.waitForFunction(()=>document.querySelectorAll('.comparison-chart .bar-track > div').length===5,{},{timeout:180000});
 const comparison=await page.evaluate(({id,flow})=>window.desktop.request(`/api/analysis/${id}/flows/${flow}/comparison`),{id:accepted.analysis_id,flow:flows[1].id});
 if(!comparison.models.every(row=>row.available))throw new Error('Not all models available');
 report.models=comparison.models;
 for(const [width,height] of [[1280,720],[1440,900],[1920,1080]]){
  await app.evaluate(({BrowserWindow},{width,height})=>BrowserWindow.getAllWindows()[0].setContentSize(width,height),{width,height});await page.waitForTimeout(200);
  const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
  if(overflow)throw new Error('Comparison overflow');
  report.viewports.push({width,height,overflow});
  await page.screenshot({path:`${directory}/screenshots/five-models-${width}.png`,fullPage:true});
 }
 await page.getByRole('button',{name:'Analyze with Phase II'}).click();
 await page.getByText('The decision, with its evidence.',{exact:true}).waitFor({timeout:240000});report.manualStarts.push('Phase I → Final Decision');
 await select(flows[2].id);await page.evaluate(()=>{location.hash='/flow'});
 await page.getByRole('button',{name:'Run fresh Phase II'}).waitFor({timeout:60000});
 await page.getByRole('button',{name:'Run fresh Phase II'}).click();
 await page.getByText('The decision, with its evidence.',{exact:true}).waitFor({timeout:240000});report.manualStarts.push('Flow Detail → Final Decision');
 await page.getByRole('link',{name:'Classification',exact:true}).click();await page.waitForTimeout(4000);
 if(!page.url().endsWith('#/phase1'))throw new Error('Manual browsing redirected');
 report.passed=report.errors.length===0;
}catch(error){report.failure=String(error);process.exitCode=1;const page=await app.firstWindow();report.url=page.url();report.visibleText=await page.locator('body').innerText();await page.screenshot({path:`${directory}/screenshots/manual-failure.png`,fullPage:true})}
finally{fs.writeFileSync(`${directory}/ui-manual-report.json`,JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));await app.close()}
