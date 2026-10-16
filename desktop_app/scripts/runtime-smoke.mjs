import {_electron as electron} from 'playwright';
import {createRequire} from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
const require=createRequire(import.meta.url);
const env={...process.env,PG_PROJECT_ROOT:path.resolve('..'),PG_DEV:'0'};delete env.ELECTRON_RUN_AS_NODE;
const packaged=process.argv.includes('--packaged');
const requireLive=process.argv.includes('--require-live');
const application=await electron.launch({executablePath:packaged?path.resolve('release/win-unpacked/ProvenanceGuard ETI.exe'):require('electron'),args:packaged?[]:['.'],env,timeout:60000});
const report={inputs:[],errors:[]};
try{
 const page=await application.firstWindow();
 page.on('pageerror',error=>report.errors.push(String(error)));
 await page.getByText('Local service connected').waitFor({timeout:60000});
 for(const [filename,stage] of [['runtime/regression-inputs/ustc-raw.csv','raw_flows'],['runtime/regression-inputs/cic-raw.csv','raw_flows'],['runtime/regression-inputs/cic-features.csv','model_features'],['../data/raw/USTC-TFC2016/Benign/Facetime.pcap',null]]){
  await page.getByRole('link',{name:'New analysis',exact:true}).first().click();
  const target=path.resolve(filename);
  await application.evaluate(({dialog},target)=>{dialog.showOpenDialog=async()=>({canceled:false,filePaths:[target]})},target);
  await page.getByRole('button',{name:/Select a local traffic file/}).click();
  if(stage)await page.getByLabel('CSV input stage').selectOption(stage);
  await page.getByRole('button',{name:'Start analysis'}).click();
  await page.getByRole('link',{name:'Explore flows',exact:true}).waitFor({timeout:120000});
  await page.getByRole('link',{name:'Explore flows',exact:true}).click();
  const first=page.locator('button.table-link').first();await first.waitFor();const flow=await first.innerText();
  if(!flow.startsWith('EXT-'))throw new Error('External flow has benchmark identity');
  await first.click();
  await page.getByRole('button',{name:'Run fresh Phase II'}).click();
  await page.locator('.progress-summary .badge').filter({hasText:/^(partial|complete)$/}).waitFor({timeout:420000});
  await page.getByRole('link',{name:'Evidence retrieval',exact:true}).click();
  await page.getByText('Fresh execution',{exact:true}).waitFor();
  await page.locator('.evidence-card').first().waitFor();
  const count=await page.locator('.evidence-card').count();if(count!==10)throw new Error('Expected 10 hybrid candidates');
  const id=await page.evaluate(()=>JSON.parse(localStorage.getItem('provenanceguard-session-v1')).state.analysisId);
  const result=await page.evaluate(({id,flow})=>window.desktop.request(`/api/analysis/${id}/flows/${flow}/phase2`),{id,flow});
  if(result.mode!=='fresh_execution'||result.retrieval.length!==30||result.reranking.length!==10||result.context.evaluation_scope!=='external_runtime'||!result.context.chunk_count)throw new Error('Incorrect external Phase-II scope/context');
  if(requireLive&&(!result.decision.final_class||result.runtime_status.fusion!=='COMPLETED'||result.unavailable.length))throw new Error('Live external fusion did not complete: '+JSON.stringify(result.blockers));
  await page.getByRole('link',{name:'Final decision',exact:true}).click();
  if(result.decision.final_class)await page.getByText('FINAL CLASS',{exact:true}).waitFor();
  else await page.getByText('No final decision for this flow',{exact:true}).waitFor();
  await page.waitForTimeout(600); // Capture the settled page after its entrance animation.
  await page.screenshot({path:path.resolve(`runtime/screenshots/external-${path.basename(filename)}.png`)});
  report.inputs.push({filename,stage,flow,analysis_id:id,retrieval:result.retrieval.length,reranking:result.reranking.length,context_chunks:result.context.chunk_count,runtime_status:result.runtime_status,final_class:result.decision.final_class,blocked:result.unavailable});
 }
 await page.getByRole('link',{name:'New analysis',exact:true}).first().click();
 await application.evaluate(({dialog},target)=>{dialog.showOpenDialog=async()=>({canceled:false,filePaths:[target]})},path.resolve('runtime/regression-inputs/incompatible.csv'));
 await page.getByRole('button',{name:/Select a local traffic file/}).click();
 await page.getByText(/CIC-Darknet2020: 62 missing predictors/).waitFor();
 await page.waitForTimeout(600);
 await page.screenshot({path:path.resolve('runtime/screenshots/incompatible-input.png')});
 if(await page.getByRole('button',{name:'Start analysis'}).isEnabled())throw new Error('Incompatible input was enabled');
 report.incompatibleRejected=true;
 const outside=path.resolve('../forbidden-export-smoke.json');
 await application.evaluate(({dialog},target)=>{dialog.showSaveDialog=async()=>({canceled:false,filePath:target})},outside);
 const contained=await page.evaluate(async(id)=>{try{await window.desktop.exportAnalysis(id,'json');return false}catch{return true}},report.inputs.at(-1).analysis_id);
 if(!contained||fs.existsSync(outside))throw new Error('Export escaped desktop_app.');
 report.outsideExportBlocked=true;
 report.passed=report.errors.length===0;
 if(!report.passed)throw new Error(report.errors.join('\n'));
}catch(error){report.failure=String(error);throw error}
finally{fs.writeFileSync('runtime/runtime-smoke-report.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));await application.close()}
