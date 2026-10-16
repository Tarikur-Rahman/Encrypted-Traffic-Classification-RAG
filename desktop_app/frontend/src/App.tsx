import {HashRouter,Routes,Route} from 'react-router-dom';
import {Shell} from './layouts/Shell';
import {Overview,NewAnalysis,Progress} from './pages/Workspace';
import {Classification,FlowExplorer,FlowDetail} from './pages/Traffic';
import {Evidence,Provenance,Reasoning,Decision} from './pages/Evidence';
import {Audit} from './pages/Audit';
import {Empty} from './components/ui';
export default function App(){return <HashRouter><Routes><Route element={<Shell/>}><Route index element={<Overview/>}/><Route path="new" element={<NewAnalysis/>}/><Route path="progress" element={<Progress/>}/><Route path="phase1" element={<Classification/>}/><Route path="flows" element={<FlowExplorer/>}/><Route path="flow" element={<FlowDetail/>}/><Route path="evidence" element={<Evidence/>}/><Route path="provenance" element={<Provenance/>}/><Route path="reasoning" element={<Reasoning/>}/><Route path="decision" element={<Decision/>}/><Route path="audit" element={<Audit/>}/><Route path="*" element={<Empty title="Page not found" to="/" action="Return to Overview"/>}/></Route></Routes></HashRouter>}
