import React from 'react';
import {describe,it,expect,vi,afterEach} from 'vitest';
import {render,screen,cleanup,act} from '@testing-library/react';
import App from '../frontend/src/App';
import {useAnalysis} from '../frontend/src/stores/analysis';
vi.mock('motion/react',()=>({useReducedMotion:()=>true,motion:{div:({children}:any)=><div>{children}</div>}}));
afterEach(()=>{cleanup();useAnalysis.setState({analysisId:null,input:null,selectedFlowId:null,phase1:null,phase2:null,finalDecision:null,status:null,flow:null,error:null,workflow:'IDLE',autoNavigationActive:false});localStorage.clear()});
describe('Core routes',()=>{
 it.each([['/','Overview'],['/new','Choose your traffic input.'],['/progress','Select an analysis to begin'],['/phase1','Classification is not available yet'],['/flows','Select an analysis to begin'],['/flow','Choose a flow to inspect'],['/evidence','Select a flow and review its evidence'],['/provenance','Select a flow and review its evidence'],['/reasoning','Select a flow and review its evidence'],['/decision','Select a flow and review its evidence'],['/audit','Select an analysis to begin']])('renders %s',async(path,title)=>{
  window.location.hash=path;
  window.desktop={request:vi.fn().mockResolvedValue({status:'ready',contracts:[],evidence:{}})} as any;
  await act(async()=>{render(<App/>)});expect(screen.getByRole('heading',{name:title,exact:true})).toBeTruthy();
 });
 it('persists analysis and selected flow across store hydration',async()=>{
  useAnalysis.getState().begin('ABCDEF123456',{name:'research.csv',dataset:'CIC-Darknet2020'});
  useAnalysis.getState().select('CIC-TEST-000092');
  const saved=JSON.parse(localStorage.getItem('provenanceguard-session-v1')!);
  expect(saved.state.analysisId).toBe('ABCDEF123456');
  expect(saved.state.selectedFlowId).toBe('CIC-TEST-000092');
  await useAnalysis.persist.rehydrate();
  expect(useAnalysis.getState().selectedFlowId).toBe('CIC-TEST-000092');
  expect(saved.state.autoNavigationActive).toBeUndefined();
 });
});
