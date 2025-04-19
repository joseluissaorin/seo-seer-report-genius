
import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, FileCheck2, AlertCircle } from 'lucide-react';
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface FileUploadZoneProps {
  files: File[];
  onFilesChange: (files: File[]) => void;
}

const FileUploadZone = ({ files, onFilesChange }: FileUploadZoneProps) => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const validCsvFiles = acceptedFiles.filter(file => file.name.endsWith('.csv'));
    if (validCsvFiles.length > 0) {
      onFilesChange(validCsvFiles);
    }
  }, [onFilesChange]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv']
    },
    multiple: true
  });

  return (
    <div className="space-y-4">
      <div 
        {...getRootProps()} 
        className={cn(
          "border-2 border-dashed rounded-xl transition-all duration-200 cursor-pointer",
          "p-8 relative overflow-hidden group",
          isDragActive ? "border-purple-400 bg-purple-50" : "border-gray-200 hover:border-purple-200 hover:bg-purple-50/50",
          files.length > 0 ? "bg-purple-50/50" : "bg-white"
        )}
      >
        <input {...getInputProps()} />
        
        <div className="relative z-10 flex flex-col items-center justify-center gap-4 text-center">
          <div className={cn(
            "rounded-full p-4 transition-colors duration-200",
            isDragActive ? "bg-purple-100" : "bg-gray-50 group-hover:bg-purple-100",
            files.length > 0 && "bg-purple-100"
          )}>
            <UploadCloud className={cn(
              "h-8 w-8 transition-colors duration-200",
              isDragActive ? "text-purple-600" : "text-gray-400 group-hover:text-purple-600",
              files.length > 0 && "text-purple-600"
            )} />
          </div>

          <div className="space-y-2">
            <p className="text-sm font-medium">
              {isDragActive ? (
                <span className="text-purple-600">Drop your CSV files here</span>
              ) : (
                <span className="text-gray-700">
                  Drag and drop your CSV files or{' '}
                  <Button 
                    variant="link" 
                    className="text-purple-600 font-semibold p-0 h-auto"
                    onClick={(e) => e.stopPropagation()}
                  >
                    browse
                  </Button>
                </span>
              )}
            </p>
            <p className="text-xs text-gray-500">
              Supports GSC exports: Queries, Pages, Devices, Countries, Dates, Search Appearance
            </p>
          </div>
        </div>

        {/* Animated gradient background */}
        <div className={cn(
          "absolute inset-0 bg-gradient-to-r from-purple-100/0 via-purple-100/30 to-purple-100/0",
          "translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000",
          isDragActive && "translate-x-0"
        )} />
      </div>

      {files.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm font-medium text-gray-700">Selected files:</p>
          <div className="grid gap-2">
            {files.map((file, index) => (
              <div
                key={index}
                className="flex items-center gap-2 p-3 bg-white rounded-lg border border-purple-100 shadow-sm"
              >
                <FileCheck2 className="h-5 w-5 text-green-500 flex-shrink-0" />
                <span className="text-sm text-gray-700 truncate flex-1">{file.name}</span>
                <Tooltip>
                  <TooltipTrigger>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-8 w-8 p-0"
                      onClick={() => onFilesChange(files.filter((_, i) => i !== index))}
                    >
                      <AlertCircle className="h-4 w-4 text-gray-400 hover:text-gray-600" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Remove file</p>
                  </TooltipContent>
                </Tooltip>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUploadZone;
