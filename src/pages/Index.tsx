
import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { useToast } from "@/components/ui/use-toast";

const Index = () => {
  const [apiKey, setApiKey] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [reportUrl, setReportUrl] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      if (selectedFile.name.endsWith('.csv')) {
        setFile(selectedFile);
      } else {
        toast({
          title: "Invalid file format",
          description: "Please upload a CSV file from Google Search Console",
          variant: "destructive"
        });
      }
    }
  };

  const handleUpload = async () => {
    if (!file) {
      toast({
        title: "No file selected",
        description: "Please select a CSV file to upload",
        variant: "destructive"
      });
      return;
    }

    if (!apiKey) {
      toast({
        title: "API Key missing",
        description: "Please enter your Gemini API key",
        variant: "destructive"
      });
      return;
    }

    setIsProcessing(true);
    setProgress(0);

    // Create a FormData object to send the file and API key
    const formData = new FormData();
    formData.append("file", file);
    formData.append("api_key", apiKey);

    try {
      // Simulate progress for now (will connect to real backend later)
      const interval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 95) {
            clearInterval(interval);
            return 95;
          }
          return prev + 5;
        });
      }, 500);

      // In a real scenario, we would make an actual API call to the FastAPI backend
      // const response = await fetch("http://localhost:8000/analyze-seo", {
      //   method: "POST",
      //   body: formData,
      // });
      
      // Simulate a delay for demo purposes
      setTimeout(() => {
        clearInterval(interval);
        setProgress(100);
        
        // Mock a successful response
        setReportUrl("/sample-report.pdf");
        
        toast({
          title: "Analysis complete!",
          description: "Your SEO report has been generated successfully."
        });
        
        setIsProcessing(false);
      }, 5000);
      
    } catch (error) {
      setIsProcessing(false);
      toast({
        title: "Error processing file",
        description: "There was an error processing your file. Please try again.",
        variant: "destructive"
      });
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-purple-50 to-indigo-50 dark:from-gray-900 dark:to-gray-800">
      <header className="py-6 border-b bg-white/80 backdrop-blur-sm dark:bg-gray-950/80 border-gray-200 dark:border-gray-800">
        <div className="container flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white">
                <circle cx="12" cy="12" r="10" />
                <path d="M12 2a4.5 4.5 0 0 0 0 9 4.5 4.5 0 0 1 0 9 10 10 0 0 0 0-18Z" />
                <path d="M12 8a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z" />
                <path d="M12 21a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z" />
              </svg>
            </div>
            <h1 className="text-xl font-bold">SEO Seer</h1>
          </div>
          <div className="hidden md:flex gap-6 text-sm">
            <a href="#" className="text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white transition">How it works</a>
            <a href="#" className="text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white transition">About</a>
            <a href="#" className="text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white transition">Contact</a>
          </div>
        </div>
      </header>

      <main className="container py-12">
        <div className="max-w-3xl mx-auto">
          <div className="text-center mb-10">
            <h1 className="text-4xl font-extrabold tracking-tight mb-4 bg-gradient-to-r from-violet-700 to-indigo-600 bg-clip-text text-transparent">
              SEO Analysis Made Intelligent
            </h1>
            <p className="text-xl text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
              Upload your Google Search Console data and get AI-powered insights and actionable recommendations.
            </p>
          </div>

          <Card className="border-0 shadow-lg">
            <CardHeader>
              <CardTitle>Generate SEO Report</CardTitle>
              <CardDescription>
                Upload your Google Search Console export file to analyze your website's SEO performance
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="upload" className="w-full">
                <TabsList className="grid w-full grid-cols-2 mb-6">
                  <TabsTrigger value="upload">Upload Data</TabsTrigger>
                  <TabsTrigger value="api">API Settings</TabsTrigger>
                </TabsList>

                <TabsContent value="upload" className="space-y-4">
                  <div className="border-2 border-dashed border-gray-200 dark:border-gray-800 rounded-lg p-8 text-center">
                    <input
                      ref={fileRef}
                      type="file"
                      accept=".csv"
                      onChange={handleFileChange}
                      className="hidden"
                    />
                    <div className="mb-4">
                      <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mx-auto text-gray-400">
                        <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
                        <polyline points="14 2 14 8 20 8" />
                        <path d="M8 13h2" />
                        <path d="M8 17h2" />
                        <path d="M14 13h2" />
                        <path d="M14 17h2" />
                      </svg>
                    </div>
                    {file ? (
                      <div>
                        <p className="text-sm font-medium">Selected file:</p>
                        <p className="text-sm text-gray-500 dark:text-gray-400">{file.name}</p>
                        <Button 
                          variant="outline" 
                          size="sm" 
                          className="mt-2"
                          onClick={() => {
                            setFile(null);
                            if (fileRef.current) fileRef.current.value = '';
                          }}
                        >
                          Change File
                        </Button>
                      </div>
                    ) : (
                      <div>
                        <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">
                          Drag and drop your CSV file, or click to browse
                        </p>
                        <Button 
                          variant="outline" 
                          onClick={() => fileRef.current?.click()}
                        >
                          Select File
                        </Button>
                      </div>
                    )}
                  </div>
                  
                  {isProcessing && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Processing your data...</span>
                        <span>{progress}%</span>
                      </div>
                      <Progress value={progress} />
                    </div>
                  )}
                </TabsContent>

                <TabsContent value="api" className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="api-key">Gemini API Key</Label>
                    <Input
                      id="api-key"
                      type="password"
                      placeholder="Enter your Gemini API key"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                    />
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      Your API key is required to generate AI-powered insights and recommendations.
                    </p>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button variant="outline" onClick={() => {
                setFile(null);
                setApiKey("");
                setProgress(0);
                setReportUrl("");
                if (fileRef.current) fileRef.current.value = '';
              }}>
                Reset
              </Button>
              <Button 
                onClick={handleUpload} 
                disabled={isProcessing || !file || !apiKey}
                className="bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-700 hover:to-indigo-700"
              >
                {isProcessing ? "Processing..." : "Generate Report"}
              </Button>
            </CardFooter>
          </Card>
          
          {reportUrl && (
            <Card className="mt-8 border-0 shadow-lg">
              <CardHeader>
                <CardTitle>Your SEO Report</CardTitle>
                <CardDescription>
                  Your report has been generated successfully
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg mb-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2 text-indigo-600">
                        <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
                        <polyline points="14 2 14 8 20 8" />
                      </svg>
                      <span>SEO Analysis Report.pdf</span>
                    </div>
                    <Button variant="ghost" size="sm" asChild>
                      <a href={reportUrl} download>
                        Download
                      </a>
                    </Button>
                  </div>
                </div>
                <div className="grid gap-4">
                  <div className="p-4 border rounded-lg">
                    <h3 className="font-medium mb-2">Report Summary</h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      The report includes a comprehensive analysis of your website's SEO performance, 
                      including traffic trends, keyword performance, and actionable recommendations.
                    </p>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 border rounded-lg">
                      <h3 className="font-medium mb-2">Top Keywords</h3>
                      <ul className="text-sm space-y-1">
                        <li className="text-gray-500 dark:text-gray-400">organic traffic optimization</li>
                        <li className="text-gray-500 dark:text-gray-400">seo strategies 2023</li>
                        <li className="text-gray-500 dark:text-gray-400">google ranking factors</li>
                      </ul>
                    </div>
                    <div className="p-4 border rounded-lg">
                      <h3 className="font-medium mb-2">Content Opportunities</h3>
                      <ul className="text-sm space-y-1">
                        <li className="text-gray-500 dark:text-gray-400">backlink building guide</li>
                        <li className="text-gray-500 dark:text-gray-400">technical seo audit</li>
                        <li className="text-gray-500 dark:text-gray-400">core web vitals optimization</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        <div className="mt-16 max-w-4xl mx-auto">
          <h2 className="text-2xl font-bold text-center mb-8">How It Works</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-12 h-12 bg-violet-100 dark:bg-violet-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-violet-600 dark:text-violet-400">
                  <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
                  <polyline points="14 2 14 8 20 8" />
                </svg>
              </div>
              <h3 className="font-medium mb-2">1. Upload GSC Data</h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Export your data from Google Search Console and upload the CSV file
              </p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-indigo-100 dark:bg-indigo-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-indigo-600 dark:text-indigo-400">
                  <circle cx="12" cy="12" r="10" />
                  <path d="M12 2a4.5 4.5 0 0 0 0 9 4.5 4.5 0 0 1 0 9 10 10 0 0 0 0-18Z" />
                  <path d="M12 8a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z" />
                  <path d="M12 21a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z" />
                </svg>
              </div>
              <h3 className="font-medium mb-2">2. AI Analysis</h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Our advanced algorithm analyzes your data with AI to extract insights
              </p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-600 dark:text-blue-400">
                  <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
                  <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
                </svg>
              </div>
              <h3 className="font-medium mb-2">3. Get Actionable Insights</h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Receive a detailed PDF report with actionable recommendations
              </p>
            </div>
          </div>
        </div>
      </main>

      <footer className="border-t border-gray-200 dark:border-gray-800 py-8 mt-12">
        <div className="container">
          <div className="flex flex-col md:flex-row items-center justify-between">
            <div className="flex items-center gap-2 mb-4 md:mb-0">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white">
                  <circle cx="12" cy="12" r="10" />
                  <path d="M12 2a4.5 4.5 0 0 0 0 9 4.5 4.5 0 0 1 0 9 10 10 0 0 0 0-18Z" />
                </svg>
              </div>
              <span className="font-semibold">SEO Seer</span>
            </div>
            <div className="text-sm text-gray-500 dark:text-gray-400">
              © {new Date().getFullYear()} SEO Seer. All rights reserved.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
