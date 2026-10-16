// NSIS portable wrappers do not reliably forward Electron's inspector pipe.
// Attach to the real renderer over a loopback debugging port instead.
import {chromium} from 'playwright';
import {spawn} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import net from 'node:net';
const listener=net.createServer();await new Promise(resolve=>listener.listen(0,'127.0.0.1',resolve));
const port=listener.address().port;await new Promise(resolve=>listener.close(resolve));
const env={...process.env};delete env.PG_PROJECT_ROOT;delete env.PG_DEV;delete env.ELECTRON_RUN_AS_NODE;
const child=spawn(path.resolve('release/ProvenanceGuard-ETI-1.0.0.exe'),[`--remote-debugging-port=${port}`],{env,windowsHide:true,stdio:'ignore'});
const report={actualPortable:true,projectRootOverride:false,passed:false,viewports:[],navigation:[],errors:[]};
let browser;
try{
 const deadline=Date.now()+180000;
 while(Date.now()<deadline){try{browser=await chromium.connectOverCDP(`http://127.0.0.1:${port}`,{timeout:1000});break}catch{await new Promise(r=>setTimeout(r,500))}}
 if(!browser)throw new Error('Portable renderer debugging endpoint did not start');
 const context=browser.contexts()[0];let page=context.pages()[0];if(!page)page=await context.waitForEvent('page',{timeout:60000});
 page.on('pageerror',error=>report.errors.push(String(error)));
 await page.getByText('Local service connected').waitFor({timeout:60000});
 const health=await page.evaluate(()=>window.desktop.request('/api/health'));
 if(path.resolve(health.project_root).toLowerCase()!==path.resolve('..').toLowerCase())throw new Error('Portable root resolution differs');
 report.resolvedProjectRoot=health.project_root;
 await page.getByRole('link',{name:'New analysis',exact:true}).first().click();
 await page.getByRole('button',{name:'Completed held-out research results'}).last().click();
 await page.getByText('Compatible research input',{exact:true}).waitFor();
 await page.getByRole('button',{name:'Start analysis'}).click();
 await page.getByText('Select a flow to continue with Provenance-Aware RAG',{exact:true}).waitFor({timeout:120000});
 await page.getByRole('radio',{name:/^Select /}).first().check();
 await page.getByText('The decision, with its evidence.',{exact:true}).waitFor({timeout:240000});
 for(const [name,heading] of [['Classification','Traffic, classified.'],['Evidence retrieval','A question grounded in traffic.'],['Context & reasoning','Evidence in. A bounded explanation out.']]){
  await page.getByRole('link',{name,exact:true}).click();await page.getByText(heading,{exact:true}).waitFor();
  await page.waitForTimeout(1200);report.navigation.push(name);
 }
 report.viewportMethod='Renderer viewport emulation; Electron does not expose Browser.setWindowBounds';
 for(const [width,height] of [[1280,720],[1440,900],[1920,1080]]){
  await page.setViewportSize({width,height});await page.waitForTimeout(300);
  const size=await page.evaluate(()=>({width:innerWidth,height:innerHeight,overflow:document.documentElement.scrollWidth>innerWidth}));
  if(size.overflow||size.width!==width||size.height!==height)throw new Error(`Portable viewport mismatch: ${JSON.stringify(size)}`);
  report.viewports.push(size);
 }
 report.passed=report.errors.length===0;
}catch(error){report.failure=String(error);process.exitCode=1}
finally{
 fs.writeFileSync('runtime/portable-smoke-report.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
 await browser?.close();
 if(child.pid)await new Promise(resolve=>{const stop=spawn('taskkill',['/PID',String(child.pid),'/T','/F'],{windowsHide:true,stdio:'ignore'});stop.on('exit',resolve)});
}
