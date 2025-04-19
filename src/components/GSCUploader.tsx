
import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, Check, AlertCircle } from 'lucide-react';
import { toast } from 'sonner';
import { ReportData } from '../types';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';

interface GSCUploaderProps {
  onUploadComplete: (data: ReportData) => void;
}

export const GSCUploader: React.FC<GSCUploaderProps> = ({ onUploadComplete }) => {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const csvFiles = acceptedFiles.filter(file => file.name.endsWith('.csv'));
    if (csvFiles.length === 0) {
      toast.error('Please upload CSV files only');
      return;
    }
    setFiles(prev => [...prev, ...csvFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv']
    }
  });

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      toast.error('Please add files to upload');
      return;
    }

    setUploading(true);
    setUploadProgress(0);

    // Simulate processing files
    const totalFiles = files.length;
    let processedFiles = 0;

    for (const file of files) {
      // Simulate file processing time
      await new Promise(resolve => setTimeout(resolve, 500));
      processedFiles++;
      setUploadProgress(Math.floor((processedFiles / totalFiles) * 100));
    }

    // Mock data for demonstration
    const mockData: ReportData = {
      queries: Array(files.length * 10).fill(0).map((_, i) => ({
        query: `Sample Query ${i + 1}`,
        clicks: Math.floor(Math.random() * 100),
        impressions: Math.floor(Math.random() * 1000),
        ctr: Math.random() * 10,
        position: Math.random() * 10 + 1
      })),
      performance: {
        clicks: Math.floor(Math.random() * 5000),
        impressions: Math.floor(Math.random() * 50000),
        ctr: Math.random() * 5,
        position: Math.random() * 10 + 1
      }
    };

    toast.success(`Successfully processed ${files.length} files`);
    setUploading(false);
    setFiles([]);
    onUploadComplete(mockData);
  };

  return (
    <div className="space-y-4">
      <div 
        {...getRootProps()} 
        className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors
          ${isDragActive ? 'border-primary bg-primary/5' : 'border-muted-foreground/25 hover:border-primary/50'}`}
      >
        <input {...getInputProps()} />
        <UploadCloud className="mx-auto h-12 w-12 text-muted-foreground" />
        <p className="mt-2 text-sm text-muted-foreground">
          {isDragActive ? 'Drop the files here' : 'Drag and drop GSC CSV files, or click to select files'}
        </p>
      </div>

      {files.length > 0 && (
        <Card>
          <CardContent className="p-4">
            <div className="text-sm font-medium mb-2">Files to upload ({files.length})</div>
            <ul className="space-y-2 mb-4">
              {files.map((file, index) => (
                <li key={index} className="flex items-center justify-between bg-muted p-2 rounded">
                  <div className="flex items-center">
                    <Check className="h-4 w-4 text-green-500 mr-2" />
                    <span className="text-sm truncate max-w-[200px]">{file.name}</span>
                    <span className="text-xs text-muted-foreground ml-2">
                      ({(file.size / 1024).toFixed(1)} KB)
                    </span>
                  </div>
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    onClick={() => removeFile(index)}
                    disabled={uploading}
                  >
                    Remove
                  </Button>
                </li>
              ))}
            </ul>

            {uploading ? (
              <div className="space-y-2">
                <Progress value={uploadProgress} className="h-2" />
                <p className="text-sm text-center text-muted-foreground">
                  Processing files: {uploadProgress}%
                </p>
              </div>
            ) : (
              <Button 
                onClick={handleUpload} 
                className="w-full"
              >
                Upload and Process {files.length} Files
              </Button>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};
