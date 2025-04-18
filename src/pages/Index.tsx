
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { useToast } from "@/components/ui/use-toast";
import { 
  Loader2, 
  UploadCloud, 
  FileText, 
  Key, 
  FileCheck2, 
  TrendingUp, 
  BarChart3, 
  LineChart, 
  PieChart, 
  Network, 
  Search, 
  LucideIcon, 
  ThumbsUp
} from "lucide-react";
import { Separator } from "@/components/ui/separator";
import { Progress } from "@/components/ui/progress";

interface FeatureCardProps {
  title: string;
  description: string;
  icon: LucideIcon;
}

const FeatureCard = ({ title, description, icon: Icon }: FeatureCardProps) => (
  <Card className="h-full">
    <CardHeader className="pb-2">
      <div className="flex items-center gap-2 mb-2">
        <div className="p-2 rounded-full bg-purple-100">
          <Icon className="h-5 w-5 text-purple-700" />
        </div>
        <CardTitle className="text-lg">{title}</CardTitle>
      </div>
    </CardHeader>
    <CardContent>
      <CardDescription className="text-sm">{description}</CardDescription>
    </CardContent>
  </Card>
);

export default function Index() {
  const [file, setFile] = useState<File | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [reportUrl, setReportUrl] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const { toast } = useToast();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (selectedFile.name.endsWith('.csv')) {
        setFile(selectedFile);
        toast({
          title: "File selected",
          description: `${selectedFile.name} ready for analysis.`,
        });
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
    setProgress(0);

    const progressInterval = setInterval(() => {
      setProgress(prev => {
        // Artificial progress that slows down as it approaches 90%
        if (prev < 20) return prev + 5;
        if (prev < 40) return prev + 4;
        if (prev < 60) return prev + 3;
        if (prev < 75) return prev + 2;
        if (prev < 85) return prev + 1;
        if (prev < 90) return prev + 0.5;
        return prev;
      });
    }, 500);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('api_key', apiKey);

      toast({
        title: "Analysis started",
        description: "We're generating your comprehensive SEO report...",
      });

      const response = await fetch('http://localhost:8000/analyze-seo', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate SEO report');
      }

      // Complete the progress bar
      setProgress(100);

      // Get the blob from the response
      const blob = await response.blob();
      
      // Create a URL for the blob
      const url = URL.createObjectURL(blob);
      setReportUrl(url);
      
      toast({
        title: "Report generated successfully! 🎉",
        description: "Your SEO analysis report is ready to download",
        variant: "default",
      });
    } catch (error) {
      clearInterval(progressInterval);
      console.error('Error:', error);
      setProgress(0);
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
    <div className="min-h-screen bg-gradient-to-b from-white to-purple-50">
      <div className="container mx-auto px-4 py-12">
        <div className="text-center mb-10">
          <h1 className="text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-700 to-purple-400 mb-4">
            SEO Seer
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Advanced SEO Analysis & Insights powered by AI
          </p>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          <div className="lg:col-span-8 lg:col-start-3">
            <Tabs defaultValue="upload" className="w-full">
              <TabsList className="grid grid-cols-2 mb-8 w-full">
                <TabsTrigger value="upload" className="py-3">
                  <div className="flex items-center gap-2">
                    <UploadCloud className="h-4 w-4" />
                    <span>Upload & Analyze</span>
                  </div>
                </TabsTrigger>
                <TabsTrigger value="about" className="py-3">
                  <div className="flex items-center gap-2">
                    <Search className="h-4 w-4" />
                    <span>About SEO Seer</span>
                  </div>
                </TabsTrigger>
              </TabsList>
              
              <TabsContent value="upload">
                <Card className="border border-purple-100 shadow-lg">
                  <CardHeader className="bg-gradient-to-r from-purple-50 to-purple-100 rounded-t-lg pb-4">
                    <CardTitle className="text-2xl text-purple-800">Generate SEO Analysis Report</CardTitle>
                    <CardDescription className="text-gray-600">
                      Upload your Google Search Console export and get comprehensive insights powered by advanced analytics and AI
                    </CardDescription>
                  </CardHeader>
                  
                  <CardContent className="pt-6">
                    <form onSubmit={handleSubmit} className="space-y-6">
                      <div className="space-y-2">
                        <Label htmlFor="file" className="text-gray-700 font-medium">Search Console CSV Export</Label>
                        <div 
                          className={`border-2 border-dashed rounded-md p-8 transition-colors duration-200 cursor-pointer flex flex-col items-center justify-center ${
                            file 
                              ? 'bg-purple-50 border-purple-200 hover:bg-purple-100' 
                              : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                          }`} 
                          onClick={() => document.getElementById('file')?.click()}
                        >
                          <div className="text-center space-y-3">
                            <div className={`rounded-full p-3 mx-auto ${file ? 'bg-purple-100' : 'bg-gray-100'}`}>
                              <UploadCloud className={`h-8 w-8 ${file ? 'text-purple-500' : 'text-gray-400'}`} />
                            </div>
                            <div className="space-y-1">
                              <Label htmlFor="file" className="text-primary font-medium cursor-pointer">
                                {file ? 'Change file' : 'Click to upload'}
                              </Label>
                              <p className="text-gray-500 text-sm">or drag and drop</p>
                            </div>
                            <p className="text-xs text-gray-500">
                              Export CSV from Google Search Console with Query, Clicks, Impressions, CTR & Position columns
                            </p>
                            {file && (
                              <div className="mt-2 flex items-center justify-center gap-2 p-3 bg-purple-100 rounded-md text-purple-700 font-medium">
                                <FileCheck2 className="h-5 w-5 text-green-500" />
                                <span>{file.name}</span>
                              </div>
                            )}
                          </div>
                          <Input id="file" type="file" accept=".csv" onChange={handleFileChange} className="hidden" />
                        </div>
                      </div>
                      
                      <div className="space-y-2">
                        <Label htmlFor="apiKey" className="text-gray-700 font-medium">Gemini API Key</Label>
                        <div className="space-y-1">
                          <div className="flex items-center space-x-2">
                            <div className="relative flex-1">
                              <Key className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                              <Input 
                                id="apiKey"
                                type="password"
                                placeholder="Enter your Gemini API key"
                                value={apiKey}
                                onChange={(e) => setApiKey(e.target.value)}
                                className="pl-10 border-gray-200 focus:border-purple-300 focus:ring focus:ring-purple-200 focus:ring-opacity-50"
                              />
                            </div>
                            <a 
                              href="https://makersuite.google.com/app/apikey" 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="text-purple-600 hover:text-purple-800 text-sm whitespace-nowrap font-medium"
                            >
                              Get API key
                            </a>
                          </div>
                          <p className="text-xs text-gray-500">
                            Your API key is only used for processing and is not stored on our servers.
                          </p>
                        </div>
                      </div>
                      
                      {isSubmitting && (
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm text-gray-500 mb-1">
                            <span>Analyzing your data...</span>
                            <span>{progress.toFixed(0)}%</span>
                          </div>
                          <Progress value={progress} className="h-2" />
                          <p className="text-xs text-gray-500 italic text-center">
                            Please wait while we run our advanced analysis algorithms
                          </p>
                        </div>
                      )}
                      
                      <Alert className="bg-amber-50 border-amber-200">
                        <ThumbsUp className="h-4 w-4 text-amber-500" />
                        <AlertTitle className="text-amber-800 font-medium">What will you get?</AlertTitle>
                        <AlertDescription className="text-amber-700">
                          A comprehensive PDF report with visualizations, keyword opportunities, content suggestions, competitor analysis, and actionable recommendations.
                        </AlertDescription>
                      </Alert>
                    </form>
                  </CardContent>
                  
                  <CardFooter className="bg-gray-50 rounded-b-lg flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-3 p-6">
                    <Button 
                      type="button" 
                      onClick={handleSubmit} 
                      disabled={isSubmitting} 
                      className="w-full bg-gradient-to-r from-purple-700 to-purple-500 hover:from-purple-800 hover:to-purple-600 text-white font-medium"
                    >
                      {isSubmitting ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Generating report...
                        </>
                      ) : (
                        <>Generate Advanced SEO Report</>
                      )}
                    </Button>
                    
                    {reportUrl && (
                      <Button 
                        variant="outline" 
                        onClick={() => window.open(reportUrl)} 
                        className="w-full border-purple-200 text-purple-700 hover:bg-purple-50"
                      >
                        <FileText className="mr-2 h-4 w-4" />
                        Download Report
                      </Button>
                    )}
                  </CardFooter>
                </Card>
              </TabsContent>
              
              <TabsContent value="about">
                <Card className="border border-purple-100 shadow-lg">
                  <CardHeader className="bg-gradient-to-r from-purple-50 to-purple-100 rounded-t-lg">
                    <CardTitle className="text-2xl text-purple-800">About SEO Seer</CardTitle>
                    <CardDescription className="text-gray-600">
                      Transforming your Google Search Console data into actionable SEO insights
                    </CardDescription>
                  </CardHeader>
                  
                  <CardContent className="space-y-8 pt-6">
                    <div>
                      <h3 className="text-xl font-semibold mb-3 text-purple-800">What is SEO Seer?</h3>
                      <p className="text-gray-600 leading-relaxed">
                        SEO Seer is an advanced analytics platform that transforms Google Search Console exports 
                        into comprehensive SEO reports with actionable insights. Using sophisticated algorithms and 
                        AI-powered analysis, it identifies optimization opportunities, provides content recommendations, 
                        and helps you outperform competitors in search results.
                      </p>
                    </div>
                    
                    <Separator className="bg-purple-100" />
                    
                    <div className="space-y-6">
                      <h3 className="text-xl font-semibold text-purple-800">Advanced Features</h3>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <FeatureCard 
                          title="AI-Powered Insights" 
                          description="Gemini AI analyzes your data to provide intelligent, actionable recommendations."
                          icon={TrendingUp}
                        />
                        <FeatureCard 
                          title="Advanced Visualizations" 
                          description="Interactive charts and graphs make complex data easy to understand."
                          icon={BarChart3}
                        />
                        <FeatureCard 
                          title="Keyword Clustering" 
                          description="Discover related keyword groups to optimize your content strategy."
                          icon={Network}
                        />
                        <FeatureCard 
                          title="Opportunity Analysis" 
                          description="Identify high-potential keywords with low CTR for quick wins."
                          icon={LineChart}
                        />
                        <FeatureCard 
                          title="Content Suggestions" 
                          description="Get data-driven ideas for new content based on search patterns."
                          icon={FileText}
                        />
                        <FeatureCard 
                          title="Competitor Analysis" 
                          description="Understand your position relative to competitors in search results."
                          icon={PieChart}
                        />
                      </div>
                    </div>
                    
                    <Separator className="bg-purple-100" />
                    
                    <div>
                      <h3 className="text-xl font-semibold mb-3 text-purple-800">How to Use</h3>
                      <ol className="list-decimal pl-5 space-y-3 text-gray-600">
                        <li>Export performance data from Google Search Console (Query, Clicks, Impressions, CTR, Position)</li>
                        <li>Upload the CSV file using the form on the upload tab</li>
                        <li>Provide your Gemini API key for AI-powered analysis</li>
                        <li>Click "Generate SEO Report" and wait for processing</li>
                        <li>Download your comprehensive PDF report with actionable insights</li>
                      </ol>
                    </div>
                    
                    <div className="bg-purple-50 p-6 rounded-xl border border-purple-100 mt-4">
                      <h4 className="font-semibold text-purple-800 mb-2">Get More From Your Search Data</h4>
                      <p className="text-gray-600">
                        SEO Seer goes beyond what Google Search Console provides, applying advanced statistical models, 
                        machine learning techniques, and computational linguistics to reveal hidden opportunities in your data.
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>
    </div>
  );
}
