import {useState} from 'react';
import {useNavigate} from 'react-router-dom';
import {ArrowRight} from 'lucide-react';
import {startManualPhase2} from '../services/phase2Execution';
import {useAnalysis} from '../stores/analysis';
import {ErrorBox} from './ui';

export function StartPhase2Button(){
 const {selectedFlowId,status,phase2Starting}=useAnalysis();const navigate=useNavigate();const [error,setError]=useState('');
 async function run(){setError('');try{await startManualPhase2();if(useAnalysis.getState().autoNavigationActive)navigate('/progress')}catch(error){setError(String(error))}}
 return <><button className="button violet" disabled={!selectedFlowId||status?.status==='running'||phase2Starting} onClick={run}>Analyze with Phase II <ArrowRight size={16}/></button><ErrorBox message={error}/></>
}
