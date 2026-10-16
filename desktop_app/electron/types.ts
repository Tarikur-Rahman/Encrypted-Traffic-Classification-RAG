export interface TrafficFile { name:string; path:string; extension:string; size:number; }
export interface DesktopBridge {
 selectTrafficFile(kind?:'csv'|'pcap'):Promise<TrafficFile|null>;
 selectExportLocation(format:'json'|'csv'):Promise<string|null>;
 exportAnalysis(analysisId:string,format:'json'|'csv'):Promise<string|null>;
 getAppVersion():Promise<string>;
 openExternalDocument(url:string):Promise<void>;
 request<T=unknown>(path:string,method?:'GET'|'POST',body?:unknown):Promise<T>;
}
