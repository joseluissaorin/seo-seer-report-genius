import React from 'react';
import { Toaster } from '@/components/ui/sonner';
import Footer from './components/Footer';
import { GSCUploader } from './components/GSCUploader';
import { GSCDownload } from './components/GSCDownload';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { SeerReport } from './components/SeerReport';
import { useState } from 'react';
import { ReportData } from './types';

const App: React.FC = () => {
  const [reportData, setReportData] = useState<ReportData | null>(null);

  return (
    <div className="flex flex-col min-h-screen">
      <main className="flex-grow">
        <div className="container mx-auto p-4">
          <h1 className="text-3xl font-semibold mb-4">SEO Seer Report Genius</h1>
          <Tabs defaultvalue="upload">
            <TabsList>
              <TabsTrigger value="upload">1. Upload GSC Data</TabsTrigger>
              <TabsTrigger value="report" disabled={!reportData}>2. View Report</TabsTrigger>
            </TabsList>
            <TabsContent value="upload">
              <p className='mb-4'>
                Upload your Google Search Console (GSC) data to generate an SEO report.
                You can upload multiple CSV files at once.
              </p>
              <GSCUploader onUploadComplete={(data: ReportData) => {
                setReportData(data);
              }} />
              <GSCDownload />
            </TabsContent>
            <TabsContent value="report">
              {reportData ? (
                <SeerReport reportData={reportData} />
              ) : (
                <p>No data uploaded yet. Please upload your GSC data to view the report.</p>
              )}
            </TabsContent>
          </Tabs>
        </div>
      </main>
      <Footer />
      <Toaster />
    </div>
  );
};

export default App;
