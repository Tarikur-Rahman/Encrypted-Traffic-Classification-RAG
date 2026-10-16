import type {DesktopBridge,TrafficFile} from '../../../electron/types';
declare global {interface Window {desktop:DesktopBridge}}
export type {TrafficFile};
export type RecordData = Record<string,any>;
export interface Flow {id:string;predicted_class:string;confidence:number;probabilities:Record<string,number>;model:string;model_run:string;source_split:string;source_row_index:number;phase2_completed?:boolean;representation?:RecordData;provenance?:RecordData;query?:RecordData;metadata?:RecordData}
export interface Stage {id:string;label:string;status:string;message:string}
export interface Status {analysis_id:string;input:RecordData;status:string;stages:Stage[];error:string|null;selected_flow_id?:string|null}
export interface Summary {total_flows:number;mean_confidence:number;class_count:number;class_distribution:{label:string;count:number}[];confidence_distribution:{label:string;count:number}[];audit:RecordData;scope:string;model:string;model_run:string;mode:string}
export interface Health {status:string;project_root:string;contracts:RecordData[];evidence:RecordData;execution:string}
