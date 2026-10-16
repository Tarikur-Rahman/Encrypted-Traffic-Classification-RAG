import {_electron as electron} from 'playwright';
import {createRequire} from 'node:module';
import path from 'node:path';
const require=createRequire(import.meta.url);
const root=path.resolve('..');
const environment={...process.env,PG_DEV:'0',PG_PROJECT_ROOT:root};
delete environment.ELECTRON_RUN_AS_NODE;
const application=await electron.launch({executablePath:path.resolve('release/win-unpacked/ProvenanceGuard ETI.exe'),args:[],env:environment,timeout:60000});
try{
 const page=await application.firstWindow();
 await page.getByText('Local service connected').waitFor({timeout:30000});
 await new Promise(resolve=>setTimeout(resolve,30000));
 console.log(JSON.stringify({passed:true,idle_seconds:30,url:page.url()}));
}finally{await application.close()}
