import {_electron as electron} from 'playwright';
import {createRequire} from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
const require=createRequire(import.meta.url);
const accepted=JSON.parse(fs.readFileSync('runtime/final-v5-external-acceptance.json','utf8')).inputs.find(r=>r.label==='CIC CSV');
const directory='runtime/phase1_auto_audit';fs.mkdirSync(`${directory}/screenshots`,{recursive:true});
const env={...process.env,PG_DEV:'0',PG_PROJECT_ROOT:path.resolve('..')};delete env.ELECTRON_RUN_AS_NODE;
const app=await electron.launch({executablePath:require('electron'),args:['.'],env,timeout:60000});
const report={passed:false,pages:[],errors:[]};
try{
 const page=await app.firstWindow();page.on('pageerror',e=>report.errors.push(String(e)));
 await page.getByText('Local service connected').waitFor({timeout:60000});
 const flows=await page.evaluate(async id=>(await window.desktop.request(`/api/analysis/${id}/flows`)).rows,accepted.analysis_id);
 async function select(flow,id=accepted.analysis_id){
  await page.evaluate(async({id,flow})=>{
   const status=await window.desktop.request(`/api/analysis/${id}/status`);
   localStorage.setItem('provenanceguard-session-v1',JSON.stringify({state:{analysisId:id,selectedFlowId:flow,input:status.input},version:0}));
  },{id,flow});
  await page.reload();await page.getByText('Local service connected').waitFor({timeout:60000});
 }
 await select(null);await page.evaluate(()=>{location.hash='/phase1'});
 await page.getByText('Mean prediction confidence by model',{exact:true}).waitFor({timeout:60000});
 await page.waitForFunction(()=>document.querySelectorAll('.aggregate-chart .recharts-bar-rectangle').length===5);
 report.noSelectionRequired=true;
 report.aggregate=await page.evaluate(id=>window.desktop.request(`/api/analysis/${id}/comparison`),accepted.analysis_id);
 if(report.aggregate.single_flow!==null||report.aggregate.models.some(m=>m.flows_evaluated!==flows.length))throw new Error('Invalid multi-flow aggregate');
 const bridgeCount=()=>fs.readdirSync('runtime/gpu_manual_audit/bridge').filter(file=>{try{return JSON.parse(fs.readFileSync(`runtime/gpu_manual_audit/bridge/${file}`,'utf8')).request.analysis_id===accepted.analysis_id}catch{return false}}).length;
 const before=bridgeCount();
 await select(flows[0].id);
 for(const route of ['phase1','flow','flows','progress','evidence','provenance','reasoning','decision','audit','','new']){
  await page.evaluate(route=>{location.hash='/'+route},route);await page.waitForTimeout(700);
  for(const [width,height] of [[1280,720],[1440,900],[1920,1080]]){
   await app.evaluate(({BrowserWindow},{width,height})=>BrowserWindow.getAllWindows()[0].setContentSize(width,height),{width,height});await page.waitForTimeout(100);
   const check=await page.evaluate(()=>({overflow:document.documentElement.scrollWidth>innerWidth,smallText:[...document.querySelectorAll('main p,main td,main th,main small,nav a')].filter(e=>e.getBoundingClientRect().width&&parseFloat(getComputedStyle(e).fontSize)<13).map(e=>e.textContent.slice(0,60))}));
   report.pages.push({route,width,height,...check});
   if(check.overflow||check.smallText.length)throw new Error(`Layout/typography failure: ${route} ${width} ${JSON.stringify(check)}`);
   if(['phase1','flow','decision'].includes(route))await page.screenshot({path:`${directory}/screenshots/${route}-${width}.png`,fullPage:true});
  }
 }
 if(bridgeCount()!==before)throw new Error('Browsing caused duplicate GPU inference');
 report.cacheReused=true;
 await select(flows[1].id);await page.evaluate(()=>{location.hash='/phase1'});
 await page.getByRole('button',{name:'Analyze with Phase II'}).click();
 await page.getByText('The decision, with its evidence.',{exact:true}).waitFor({timeout:240000});report.manualPhase1Final=true;
 await select(flows[2].id);await page.evaluate(()=>{location.hash='/flow'});
 await page.getByRole('button',{name:'Run fresh Phase II'}).click();
 await page.getByText('The decision, with its evidence.',{exact:true}).waitFor({timeout:240000});report.manualFlowFinal=true;
 await page.getByRole('link',{name:'Classification',exact:true}).click();await page.waitForTimeout(3000);
 if(!page.url().endsWith('#/phase1'))throw new Error('Manual browsing redirected');
 report.manualBrowsing=true;
 const single=JSON.parse(fs.readFileSync(`${directory}/single-flow-report.json`,'utf8'));
 await select(null,single.analysis_id);await page.evaluate(()=>{location.hash='/phase1'});
 await page.getByText('Current-flow prediction confidence',{exact:true}).waitFor({timeout:60000});
 await page.waitForFunction(()=>document.querySelectorAll('.comparison-chart .bar-track > div').length===5);
 report.singleFlowExact=true;
 await page.screenshot({path:`${directory}/screenshots/single-flow.png`,fullPage:true});
 report.passed=report.errors.length===0;
}catch(error){report.failure=String(error);process.exitCode=1;const page=await app.firstWindow();report.text=await page.locator('body').innerText();await page.screenshot({path:`${directory}/screenshots/failure.png`,fullPage:true})}
finally{fs.writeFileSync(`${directory}/ui-report.json`,JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));await app.close()}
