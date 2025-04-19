
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FileText } from "lucide-react";

export const GSCDownload = () => {
  const downloadSampleData = () => {
    // For demo purposes, you can download a sample CSV
    const sampleData = `date,query,page,clicks,impressions,ctr,position
2025-04-19,seo tools,/tools,100,1000,0.1,1.5
2025-04-19,seo analysis,/analysis,50,500,0.1,2.0`;

    const blob = new Blob([sampleData], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sample-gsc-data.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  return (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle>How to Get Your GSC Data</CardTitle>
        <CardDescription>Follow these steps to download your Google Search Console data</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <ol className="list-decimal ml-4 space-y-2">
          <li>Go to <a href="https://search.google.com/search-console" target="_blank" rel="noopener noreferrer" className="text-purple-600 hover:underline">Google Search Console</a></li>
          <li>Select your property</li>
          <li>Go to Performance</li>
          <li>Set your desired date range</li>
          <li>Click the Export button (↓) and select CSV</li>
        </ol>
        <div className="mt-4">
          <Button onClick={downloadSampleData} variant="outline" className="gap-2">
            <FileText className="h-4 w-4" />
            Download Sample Data
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};
