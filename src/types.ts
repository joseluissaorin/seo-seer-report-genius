
export interface ReportData {
  queries?: Record<string, any>[];
  pages?: Record<string, any>[];
  keywords?: Record<string, any>[];
  performance?: {
    clicks?: number;
    impressions?: number;
    ctr?: number;
    position?: number;
  };
}
