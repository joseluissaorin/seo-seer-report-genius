
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { useToast } from "@/components/ui/use-toast";
import { Loader2, UploadCloud, FileText, Key, FileCheck2 } from "lucide-react";
import { Separator } from "@/components/ui/separator";

export default function Index() {
  const [file, setFile] = useState<File | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [reportUrl, setReportUrl] = useState<string | null>(null);
  const { toast } = useToast();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (selectedFile.name.endsWith('.csv')) {
        setFile(selectedFile);
      } else {
        toast({
          title: "Invalid file format",
          description: "Please upload a CSV export from Google Search Console",
          variant: "destructive",
        });
        e.target.value = '';
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!file) {
      toast({
        title: "Missing file",
        description: "Please upload a CSV export from Google Search Console",
        variant: "destructive",
      });
      return;
    }

    if (!apiKey || apiKey.length < 10) {
      toast({
        title: "Invalid API key",
        description: "Please enter a valid Gemini API key",
        variant: "destructive",
      });
      return;
    }

    setIsSubmitting(true);
    setReportUrl(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('api_key', apiKey);

      const response = await fetch('http://localhost:8000/analyze-seo', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate SEO report');
      }

      // Get the blob from the response
      const blob = await response.blob();
      
      // Create a URL for the blob
      const url = URL.createObjectURL(blob);
      setReportUrl(url);
      
      toast({
        title: "Report generated successfully",
        description: "Your SEO analysis report is ready to download",
        variant: "default",
      });
    } catch (error) {
      console.error('Error:', error);
      toast({
        title: "Error generating report",
        description: error instanceof Error ? error.message : "An unknown error occurred",
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-purple-800 mb-2">SEO Seer</h1>
        <p className="text-xl text-gray-600">Advanced SEO Analysis & Insights</p>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-8 lg:col-start-3">
          <Tabs defaultValue="upload" className="w-full">
            <TabsList className="grid grid-cols-2 mb-6">
              <TabsTrigger value="upload">Upload & Analyze</TabsTrigger>
              <TabsTrigger value="about">About SEO Seer</TabsTrigger>
            </TabsList>
            
            <TabsContent value="upload">
              <Card>
                <CardHeader>
                  <CardTitle>Generate SEO Analysis Report</CardTitle>
                  <CardDescription>
                    Upload your Google Search Console export and get comprehensive insights and recommendations
                  </CardDescription>
                </CardHeader>
                
                <CardContent>
                  <form onSubmit={handleSubmit} className="space-y-6">
                    <div className="space-y-2">
                      <Label htmlFor="file">Search Console CSV Export</Label>
                      <div className="border-2 border-dashed rounded-md p-6 bg-gray-50 hover:bg-gray-100 transition-colors cursor-pointer" onClick={() => document.getElementById('file')?.click()}>
                        <div className="flex flex-col items-center justify-center space-y-2 text-center">
                          <UploadCloud className="h-8 w-8 text-purple-500" />
                          <div className="text-sm">
                            <Label htmlFor="file" className="text-primary font-medium cursor-pointer">Click to upload</Label>
                            <p className="text-gray-500">or drag and drop</p>
                          </div>
                          <p className="text-xs text-gray-500">CSV from Google Search Console</p>
                          {file && (
                            <div className="mt-2 flex items-center space-x-2 p-2 bg-purple-50 rounded-md">
                              <FileCheck2 className="h-4 w-4 text-green-500" />
                              <span className="text-sm text-gray-700">{file.name}</span>
                            </div>
                          )}
                        </div>
                        <Input id="file" type="file" accept=".csv" onChange={handleFileChange} className="hidden" />
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="apiKey">Gemini API Key</Label>
                      <div className="flex items-center space-x-2">
                        <div className="relative flex-1">
                          <Key className="absolute left-2 top-2.5 h-4 w-4 text-gray-400" />
                          <Input 
                            id="apiKey"
                            type="password"
                            placeholder="Enter your Gemini API key"
                            value={apiKey}
                            onChange={(e) => setApiKey(e.target.value)}
                            className="pl-8"
                          />
                        </div>
                        <a 
                          href="https://makersuite.google.com/app/apikey" 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-purple-600 hover:text-purple-800 text-sm whitespace-nowrap"
                        >
                          Get API key
                        </a>
                      </div>
                      <p className="text-xs text-gray-500">
                        Your API key is only used for processing and is not stored on our servers.
                      </p>
                    </div>
                    
                    <Alert className="bg-amber-50 border-amber-200">
                      <AlertTitle className="text-amber-800">What will you get?</AlertTitle>
                      <AlertDescription className="text-amber-700">
                        A comprehensive PDF report with visualizations, keyword opportunities, content suggestions, and actionable recommendations.
                      </AlertDescription>
                    </Alert>
                  </form>
                </CardContent>
                
                <CardFooter className="flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-3">
                  <Button type="button" onClick={handleSubmit} disabled={isSubmitting} className="w-full">
                    {isSubmitting ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Generating report...
                      </>
                    ) : (
                      <>Generate SEO Report</>
                    )}
                  </Button>
                  
                  {reportUrl && (
                    <Button variant="outline" onClick={() => window.open(reportUrl)} className="w-full">
                      <FileText className="mr-2 h-4 w-4" />
                      Download Report
                    </Button>
                  )}
                </CardFooter>
              </Card>
            </TabsContent>
            
            <TabsContent value="about">
              <Card>
                <CardHeader>
                  <CardTitle>About SEO Seer</CardTitle>
                  <CardDescription>
                    Transforming your Google Search Console data into actionable SEO insights
                  </CardDescription>
                </CardHeader>
                
                <CardContent className="space-y-6">
                  <div>
                    <h3 className="text-lg font-semibold mb-2">What is SEO Seer?</h3>
                    <p className="text-gray-600">
                      SEO Seer is a powerful analysis tool that transforms Google Search Console exports into comprehensive 
                      SEO reports with actionable insights. Using advanced analytics combined with AI, it provides meaningful 
                      recommendations for improving your website's search engine performance.
                    </p>
                  </div>
                  
                  <Separator />
                  
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold">Features</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="flex items-start space-x-2">
                        <div className="bg-purple-100 p-2 rounded-full mt-1">
                          <FileText className="h-4 w-4 text-purple-600" />
                        </div>
                        <div>
                          <h4 className="font-medium">Comprehensive Analysis</h4>
                          <p className="text-sm text-gray-600">Advanced SEO metrics and visualizations</p>
                        </div>
                      </div>
                      <div className="flex items-start space-x-2">
                        <div className="bg-purple-100 p-2 rounded-full mt-1">
                          <FileText className="h-4 w-4 text-purple-600" />
                        </div>
                        <div>
                          <h4 className="font-medium">AI-Powered Insights</h4>
                          <p className="text-sm text-gray-600">Gemini AI provides intelligent recommendations</p>
                        </div>
                      </div>
                      <div className="flex items-start space-x-2">
                        <div className="bg-purple-100 p-2 rounded-full mt-1">
                          <FileText className="h-4 w-4 text-purple-600" />
                        </div>
                        <div>
                          <h4 className="font-medium">Keyword Opportunities</h4>
                          <p className="text-sm text-gray-600">Discover new high-potential keywords</p>
                        </div>
                      </div>
                      <div className="flex items-start space-x-2">
                        <div className="bg-purple-100 p-2 rounded-full mt-1">
                          <FileText className="h-4 w-4 text-purple-600" />
                        </div>
                        <div>
                          <h4 className="font-medium">Content Suggestions</h4>
                          <p className="text-sm text-gray-600">Get ideas for new content based on your data</p>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <Separator />
                  
                  <div>
                    <h3 className="text-lg font-semibold mb-2">How to Use</h3>
                    <ol className="list-decimal pl-5 space-y-2 text-gray-600">
                      <li>Export performance data from Google Search Console (Query, Clicks, Impressions, CTR, Position)</li>
                      <li>Upload the CSV file using the form on the upload tab</li>
                      <li>Provide your Gemini API key for AI-powered analysis</li>
                      <li>Click "Generate SEO Report" and wait for processing</li>
                      <li>Download your comprehensive PDF report</li>
                    </ol>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}
