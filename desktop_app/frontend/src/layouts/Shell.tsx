import {NavLink,Outlet,useLocation} from 'react-router-dom';
import {Activity,ArrowUpDown,BookOpen,CheckCircle2,ChevronRight,FileClock,FlaskConical,LayoutDashboard,ListFilter,Network,Plus,ScanLine,ShieldCheck,Sparkles} from 'lucide-react';
import {motion,useReducedMotion} from 'motion/react';
import {useAnalysis} from '../stores/analysis';
import {useResearch} from '../hooks/useResearch';
import {Badge,ErrorBox,StatusBadge} from '../components/ui';

export const navigation=[
 {label:'Workspace',items:[{to:'/',label:'Overview',icon:LayoutDashboard},{to:'/new',label:'New analysis',icon:Plus},{to:'/progress',label:'Analysis progress',icon:Activity}]},
 {label:'Phase I · Traffic classification',items:[{to:'/phase1',label:'Classification',icon:ScanLine},{to:'/flows',label:'Flow explorer',icon:ListFilter},{to:'/flow',label:'Flow detail',icon:Network}]},
 {label:'Phase II · Provenance-aware RAG',items:[{to:'/evidence',label:'Evidence retrieval',icon:BookOpen},{to:'/provenance',label:'Provenance & reranking',icon:ArrowUpDown},{to:'/reasoning',label:'Context & reasoning',icon:Sparkles},{to:'/decision',label:'Final decision',icon:CheckCircle2}]},
 {label:'Research record',items:[{to:'/audit',label:'Audit & export',icon:FileClock}]},
];

export function Shell(){
 useResearch();const state=useAnalysis();const location=useLocation();const reduced=useReducedMotion();
 const title=navigation.flatMap(group=>group.items).find(item=>item.to===location.pathname)?.label||'Workspace';
 const workflowPhase=state.workflow.includes('PHASE2')||state.workflow.includes('LLM')||state.workflow.includes('FUSION')?'phase2':'phase1';
 function manualLink(event:React.MouseEvent<HTMLElement>){if((event.target as Element).closest('a[href]'))state.cancelAutoNavigation()}
 return <div className="app-shell" onClickCapture={manualLink}><aside className="sidebar"><div className="brand"><div className="brand-mark"><ShieldCheck size={22}/></div><div><strong>ProvenanceGuard</strong><span>ENCRYPTED TRAFFIC INTELLIGENCE</span></div></div><div className="workspace-label"><FlaskConical size={15}/>Research workspace <Badge>ETI</Badge></div><nav aria-label="Main navigation">{navigation.map((group,index)=><div className={`nav-group group-${index}`} key={group.label}><h2>{group.label}</h2>{group.items.map(item=><NavLink key={item.to} to={item.to} end><item.icon size={17}/><span>{item.label}</span></NavLink>)}</div>)}</nav><div className="sidebar-footer"><ShieldCheck size={17}/><div><strong>Research integrity first</strong><span>Original project stays read-only</span></div></div></aside>
 <div className="workspace"><header className="topbar"><div>Workspace <ChevronRight size={13}/><strong>{title}</strong></div><div className="topbar-status"><span className={`runtime-state ${state.health?.status==='ready'?'online':'offline'}`}><i/>{state.health?.status==='ready'?'Local service connected':'Local service offline'}</span><span className="analysis-context">{state.input?.dataset||'No dataset'} · <span className="mono">{state.analysisId||'NO ACTIVE ANALYSIS'}</span></span><StatusBadge status={state.workflow} phase={workflowPhase}/><StatusBadge status={state.llmStatus?.status==='READY'?'LLM ONLINE':'LLM OFFLINE'}/></div></header><main><ErrorBox message={state.error||''}/><motion.div key={location.pathname} initial={{opacity:0,y:reduced?0:4}} animate={{opacity:1,y:0}} transition={{duration:reduced?0:0.18}}><Outlet/></motion.div></main><footer className="bottom-bar"><span>ProvenanceGuard ETI <span className="muted">/ Research desktop</span></span><span>Local artifacts · Explicit provenance · No model training</span></footer></div></div>;
}
