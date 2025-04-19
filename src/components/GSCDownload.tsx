
import React from 'react';
import { Download, ExternalLink } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';

export const GSCDownload: React.FC = () => {
  const handleDownloadSample = () => {
    // In a real app, this would download an actual file
    toast.success('Sample file download started');
    // Simulate download
    setTimeout(() => {
      toast.info('Sample file downloaded successfully');
    }, 1500);
  };

  return (
    <Card className="mt-8">
      <CardHeader>
        <CardTitle>Don't have GSC data?</CardTitle>
        <CardDescription>
          Learn how to export your Google Search Console data or download a sample file to see how the tool works.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-4 md:grid-cols-2">
          <Button 
            variant="outline" 
            className="flex items-center" 
            onClick={() => 
              window.open(
                'https://support.google.com/webmasters/answer/7576553', 
                '_blank'
              )
            }
          >
            <ExternalLink className="mr-2 h-4 w-4" />
            How to export GSC data
          </Button>
          <Button 
            variant="secondary" 
            className="flex items-center"
            onClick={handleDownloadSample}
          >
            <Download className="mr-2 h-4 w-4" />
            Download sample data
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};
