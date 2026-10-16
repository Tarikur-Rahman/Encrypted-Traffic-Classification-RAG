import React from 'react';
import {cleanup,render,screen,waitFor} from '@testing-library/react';
import {afterEach,expect,it,vi} from 'vitest';
import {ModelComparison} from '../frontend/src/components/ModelComparison';
import {useAnalysis} from '../frontend/src/stores/analysis';
import {api} from '../frontend/src/services/api';
vi.mock('../frontend/src/services/api',()=>({api:vi.fn()}));
const names=['Random Forest','1D CNN','XGBoost','LightGBM','CNN-LSTM'];
const models=names.map((name,i)=>({name,model:String(i),retained:i===3,status:'Completed',flows_evaluated:3,mean_confidence:.8,agreement:2/3,reason:''}));
afterEach(()=>{cleanup();vi.clearAllMocks()});
it('populates all five aggregate results without selecting any flow',async()=>{
 useAnalysis.setState({analysisId:'ABCDEF123456',selectedFlowId:null});
 vi.mocked(api).mockResolvedValue({status:'Completed',total_flows:3,models,single_flow:null});
 render(<ModelComparison/>);
 await screen.findByText('Mean prediction confidence by model');
 names.forEach(name=>expect(screen.getAllByText(name).length).toBeGreaterThan(0));
 expect(screen.queryByText('Predicted class')).toBeNull();
 expect(api).toHaveBeenCalledWith('/api/analysis/ABCDEF123456/comparison');
});
it('shows exact predictions only for a single-flow analysis',async()=>{
 useAnalysis.setState({analysisId:'ABCDEF123456',selectedFlowId:null});
 vi.mocked(api).mockResolvedValue({status:'Completed',total_flows:1,models,single_flow:models.map(m=>({...m,predicted_class:'Tor',confidence:.9}))});
 render(<ModelComparison/>);await screen.findByText('Predicted class');
 expect(screen.getAllByText('Tor')).toHaveLength(5);
 expect(screen.queryByText('Mean confidence')).toBeNull();
});
it('loads the selected flow cache through a read-only request',async()=>{
 useAnalysis.setState({analysisId:'ABCDEF123456',selectedFlowId:'FLOW-N'});
 vi.mocked(api).mockResolvedValue({status:'Completed',models:models.map(m=>({...m,predicted_class:'VPN',confidence:.9}))});
 render(<ModelComparison perFlow/>);
 await waitFor(()=>expect(api).toHaveBeenCalledWith('/api/analysis/ABCDEF123456/flows/FLOW-N/comparison'));
 await screen.findByText('Current-flow prediction confidence');
 expect(vi.mocked(api).mock.calls.every(call=>call.length===1)).toBe(true);
});
