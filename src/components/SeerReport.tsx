
import React from 'react';
import { ReportData } from '../types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

interface SeerReportProps {
  reportData: ReportData;
}

export const SeerReport: React.FC<SeerReportProps> = ({ reportData }) => {
  // Check if we have query data to display
  const hasQueryData = reportData.queries && reportData.queries.length > 0;
  
  // Create chart data from the top 10 queries
  const chartData = hasQueryData 
    ? reportData.queries!.slice(0, 10).map(query => ({
        name: query.query,
        clicks: query.clicks,
        impressions: query.impressions / 10, // Scale down impressions to fit on the same chart
      }))
    : [];

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Total Clicks</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {reportData.performance?.clicks?.toLocaleString() || 0}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Total Impressions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {reportData.performance?.impressions?.toLocaleString() || 0}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Average CTR</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {reportData.performance?.ctr?.toFixed(2) || 0}%
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Average Position</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {reportData.performance?.position?.toFixed(1) || 0}
            </div>
          </CardContent>
        </Card>
      </div>

      {hasQueryData && (
        <Card className="col-span-4">
          <CardHeader>
            <CardTitle>Top Queries Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                  <XAxis 
                    dataKey="name" 
                    angle={-45} 
                    textAnchor="end" 
                    height={70} 
                    tick={{ fontSize: 12 }}
                  />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="clicks" fill="#4f46e5" name="Clicks" />
                  <Bar dataKey="impressions" fill="#84cc16" name="Impressions (÷10)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}

      {!hasQueryData && (
        <Card>
          <CardContent className="p-8 text-center">
            <p className="text-muted-foreground">
              No query data available. This could be due to missing data in the uploaded files.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
