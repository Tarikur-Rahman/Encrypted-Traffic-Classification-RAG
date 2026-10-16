import {_electron as electron} from 'playwright';
import {createRequire} from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
const require=createRequire(import.meta.url);
const output=path.resolve('runtime/screenshots');fs.mkdirSync(output,{recursive:true});
const packaged=process.argv.includes('--packaged');
const dev=process.argv.includes('--dev');
let vite;if(dev){const {createServer}=await import('vite');vite=await createServer({configFile:'frontend/vite.config.ts'});await vite.listen();}
const environment={...process.env,PG_DEV:dev?'1':'0',PG_PROJECT_ROOT:path.resolve('..')};delete environment.ELECTRON_RUN_AS_NODE;
const application=await electron.launch({executablePath:packaged?path.resolve('release/win-unpacked/ProvenanceGuard ETI.exe'):require('electron'),args:packaged?[]:['.'],env:environment,timeout:60000});
const report={packaged,dev,pages:[],errors:[],exports:[],nativePicker:[],contrast:[]};
try{
 const page=await application.firstWindow();
 page.on('pageerror',error=>report.errors.push(String(error)));
 page.on('console',msg=>{if(msg.type()==='error')report.errors.push(msg.text())});
 await page.waitForLoadState('domcontentloaded');
 await page.screenshot({path:path.join(output,'00-startup.png')});
 report.startup=await page.evaluate(()=>({url:location.href,text:document.body.innerText,bridge:typeof window.desktop}));
 await page.getByText('Local service connected').waitFor({timeout:30000});
 await page.screenshot({path:path.join(output,'01-overview.png')});
 await page.getByRole('link',{name:'New analysis',exact:true}).first().click();
 await page.getByRole('button',{name:'Completed held-out research results'}).last().click();
 await page.getByText('Compatible research input',{exact:true}).waitFor();
 await page.getByRole('button',{name:'Start analysis'}).click();
 await page.getByText('Select a flow to continue with Provenance-Aware RAG',{exact:true}).waitFor({timeout:120000});
 // Keep auto-navigation active while selecting the eligible flow. Manual
 // navigation checks run after the automatic Phase-II transition below.
 await page.screenshot({path:path.join(output,'02-flow-required.png')});
 // TEST query IDs are owned by the current V5 run; select the first
 // authoritative flow instead of depending on a retired historical ID.
 const firstFlow=page.locator('button.table-link').first();
 await firstFlow.waitFor({timeout:30000});
 await page.screenshot({path:path.join(output,'03-flow-explorer.png')});
 // Select through the native radio control as well as the visible flow link;
 // this exercises the same selection contract used by keyboard users.
 const firstSelector=page.getByRole('radio',{name:/^Select /}).first();
 await firstSelector.check();
 await page.locator('tr.selected-row').waitFor({timeout:10000});
 await page.getByText('The decision, with its evidence.',{exact:true}).waitFor({timeout:240000});
 // Once the automatic transition completes, each manual section link must
 // remain on its own destination instead of being redirected by stale state.
 for(const [label,heading] of [['Classification','Traffic, classified.'],['Evidence retrieval','A question grounded in traffic.'],['Context & reasoning','Evidence in. A bounded explanation out.']]){
  await page.getByRole('link',{name:label,exact:true}).click();
  await page.getByText(heading,{exact:true}).waitFor();
 }
 await page.getByRole('link',{name:'Final decision',exact:true}).click();
 await page.getByText('The decision, with its evidence.',{exact:true}).waitFor();
 for(const [name,file] of [['Evidence retrieval','04-evidence'],['Provenance & reranking','05-provenance'],['Context & reasoning','06-reasoning'],['Final decision','07-decision'],['Audit & export','08-audit']]){
  await page.getByRole('link',{name,exact:true}).click();
  await page.waitForTimeout(1000);
  for(const [width,height] of [[1440,900],[1280,720],[1920,1080]]){
   await application.evaluate(({BrowserWindow},{width,height})=>BrowserWindow.getAllWindows()[0].setContentSize(width,height),{width,height});
   await page.waitForTimeout(200);
   const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth);
   if(overflow)throw new Error(`Window overflow: ${name} ${width}x${height}`);
   report.pages.push({name,width,height,overflow});
  }
  await application.evaluate(({BrowserWindow})=>BrowserWindow.getAllWindows()[0].setContentSize(1440,900));
  await page.screenshot({path:path.join(output,file+'.png')});
  const contrast=await page.evaluate(()=>{
   const color=(s)=>s.match(/[\d.]+/g)?.map(Number)||[0,0,0];
   const lum=c=>c.slice(0,3).map(v=>{v/=255;return v<=.04045?v/12.92:((v+.055)/1.055)**2.4}).reduce((a,v,i)=>a+v*[.2126,.7152,.0722][i],0);
   const issues=[];let checked=0;
   for(const element of document.querySelectorAll('main h1,main h2,main h3,main p,main span,main strong,main td,main th,nav a')){
    if(!element.textContent.trim()||!element.getBoundingClientRect().width||element.children.length)continue;
    const style=getComputedStyle(element);if(Number(style.opacity)<1)continue;
    let parent=element;let bg;
    while(parent){const c=color(getComputedStyle(parent).backgroundColor);if(c.length===3||c[3]===1){bg=c;break}parent=parent.parentElement}
    if(!bg)continue;
    const fg=color(style.color);const a=lum(fg),b=lum(bg);const ratio=(Math.max(a,b)+.05)/(Math.min(a,b)+.05);
    const threshold=parseFloat(style.fontSize)>=24?3:4.5;checked++;
    if(ratio<threshold)issues.push({text:element.textContent.slice(0,70),ratio,foreground:style.color,background:bg});
   }
   return {checked,issues};
  });
  report.contrast.push({name,...contrast});
  if(contrast.issues.length)throw new Error(`Text contrast failed: ${JSON.stringify(contrast.issues)}`);
 }
 const id=await page.evaluate(()=>JSON.parse(localStorage.getItem('provenanceguard-session-v1')).state.analysisId);
 const unsafe=await page.evaluate(async()=>{try{await window.desktop.openExternalDocument('javascript:alert(1)');return false}catch{return true}});
 if(!unsafe)throw new Error('Unsafe external protocol was accepted.');
 const existing=path.resolve('runtime/no-overwrite-test.txt');fs.writeFileSync(existing,'preserve');
 await application.evaluate(({dialog},target)=>{dialog.showSaveDialog=async()=>({canceled:false,filePath:target})},existing);
 const blocked=await page.evaluate(async(id)=>{try{await window.desktop.exportAnalysis(id,'json');return false}catch{return true}},id);
 if(!blocked||fs.readFileSync(existing,'utf8')!=='preserve')throw new Error('Existing export destination was not protected.');
 report.overwriteBlocked=true;
 const outside=path.resolve('../forbidden-export-smoke.json');
 await application.evaluate(({dialog},target)=>{dialog.showSaveDialog=async()=>({canceled:false,filePath:target})},outside);
 const contained=await page.evaluate(async(id)=>{try{await window.desktop.exportAnalysis(id,'json');return false}catch{return true}},id);
 if(!contained||fs.existsSync(outside))throw new Error('Export escaped desktop_app.');
 report.outsideExportBlocked=true;
 for(const format of ['json','csv']){
  const target=path.resolve(`runtime/smoke-export-${Date.now()}.${format}`);
  await application.evaluate(({dialog},target)=>{dialog.showSaveDialog=async()=>({canceled:false,filePath:target})},target);
  const saved=await page.evaluate(({id,format})=>window.desktop.exportAnalysis(id,format),{id,format});
  if(saved!==target||!fs.existsSync(target))throw new Error('Native export did not write the selected new destination.');
  report.exports.push({format,bytes:fs.statSync(target).size});
 }
 // Exercise the real preload/native-picker bridge; only the dialog response is automated.
 for(const extension of ['csv','pcap','pcapng']){
  const target=path.resolve(`runtime/picker-test.${extension}`);fs.writeFileSync(target,'test');
  await application.evaluate(({dialog},target)=>{dialog.showOpenDialog=async()=>({canceled:false,filePaths:[target]})},target);
  const selected=await page.evaluate(()=>window.desktop.selectTrafficFile());
  if(selected.path!==target||selected.extension!=='.'+extension)throw new Error('Native picker contract failed.');
  report.nativePicker.push(extension);
 }
 await page.reload();
 await page.getByText('Local service connected').waitFor();
 const restored=await page.evaluate(()=>JSON.parse(localStorage.getItem('provenanceguard-session-v1')).state.analysisId);
 if(restored!==id)throw new Error('Analysis state was lost after navigation/reload.');
 report.sessionPersisted=true;
 report.passed=report.errors.length===0;
 fs.writeFileSync(path.resolve(packaged?'runtime/packaged-smoke-report.json':'runtime/desktop-smoke-report.json'),JSON.stringify(report,null,2));
 console.log(JSON.stringify(report,null,2));
 if(!report.passed)process.exitCode=1;
}catch(error){report.failure=String(error);fs.writeFileSync(path.resolve('runtime/desktop-smoke-report.json'),JSON.stringify(report,null,2));throw error}
finally{await application.close();await vite?.close()}
