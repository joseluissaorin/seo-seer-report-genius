
import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { useToast } from "@/components/ui/use-toast";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Progress } from "@/components/ui/progress";
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
  ThumbsUp,
  Globe2,
  Smartphone,
  Calendar,
  Layers,
  RadioTower,
  Trophy,
  Users,
  MousePointerClick,
  Microscope,
  ExternalLink,
  LucideIcon,
  Lightbulb
} from "lucide-react";
import FileUploadZone from '@/components/FileUploadZone';

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
  const [files, setFiles] = useState<File[]>([]);
  const [apiKey, setApiKey] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [reportUrl, setReportUrl] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const { toast } = useToast();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files);
      const validCsvFiles = selectedFiles.filter(file => file.name.endsWith('.csv'));

      if (validCsvFiles.length > 0) {
        setFiles(validCsvFiles);
        toast({
          title: "Files selected",
          description: `${validCsvFiles.length} CSV file(s) ready for analysis.`,
        });
      } else {
        toast({
          title: "Invalid file format",
          description: "Please upload only CSV exports from Google Search Console",
          variant: "destructive",
        });
        e.target.value = '';
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (files.length === 0) {
      toast({
        title: "Missing files",
        description: "Please upload at least one CSV export from Google Search Console",
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
      // Make sure we're passing files with the correct field name expected by the backend
      files.forEach(file => {
        formData.append('file', file);
      });
      formData.append('api_key', apiKey);

      toast({
        title: "Analysis started",
        description: `Processing ${files.length} CSV file(s)...`,
      });

      const response = await fetch('http://localhost:4568/analyze-seo', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error response:', errorText);
        throw new Error(errorText || 'Failed to generate SEO report');
      }

      setProgress(100);

      const blob = await response.blob();
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

  const navigateToUploadTab = () => {
    const uploadTab = document.querySelector('[data-state="inactive"][value="upload"]');
    if (uploadTab && uploadTab instanceof HTMLElement) {
      uploadTab.click();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-purple-50">
      <div className="container mx-auto px-4 py-12">
        <div className="text-center mb-10">
          <div className="flex items-center justify-center gap-2 mb-2">
            <TrendingUp className="h-8 w-8 text-purple-700" />
            <h1 className="text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-700 to-purple-400">
              SEO Seer Pro
            </h1>
          </div>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Enterprise-Grade SEO Analysis & Insights Platform
          </p>
          <div className="flex flex-wrap justify-center gap-2 mt-4">
            <Badge variant="secondary" className="bg-purple-100 hover:bg-purple-200 text-purple-800">Keyword Research</Badge>
            <Badge variant="secondary" className="bg-purple-100 hover:bg-purple-200 text-purple-800">Competitor Analysis</Badge>
            <Badge variant="secondary" className="bg-purple-100 hover:bg-purple-200 text-purple-800">Content Strategy</Badge>
            <Badge variant="secondary" className="bg-purple-100 hover:bg-purple-200 text-purple-800">AI-Powered Insights</Badge>
            <Badge variant="secondary" className="bg-purple-100 hover:bg-purple-200 text-purple-800">Advanced Visualizations</Badge>
          </div>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          <div className="lg:col-span-10 lg:col-start-2">
            <Tabs defaultValue="upload" className="w-full">
              <TabsList className="grid grid-cols-3 mb-8 w-full">
                <TabsTrigger value="upload" className="py-3">
                  <div className="flex items-center gap-2">
                    <UploadCloud className="h-4 w-4" />
                    <span>Upload & Analyze</span>
                  </div>
                </TabsTrigger>
                <TabsTrigger value="features" className="py-3">
                  <div className="flex items-center gap-2">
                    <Layers className="h-4 w-4" />
                    <span>Pro Features</span>
                  </div>
                </TabsTrigger>
                <TabsTrigger value="about" className="py-3">
                  <div className="flex items-center gap-2">
                    <Search className="h-4 w-4" />
                    <span>About SEO Seer Pro</span>
                  </div>
                </TabsTrigger>
              </TabsList>
              
              <TabsContent value="upload">
                <Card className="border border-purple-100 shadow-lg">
                  <CardHeader className="bg-gradient-to-r from-purple-50 to-purple-100 rounded-t-lg pb-4">
                    <CardTitle className="text-2xl text-purple-800">Generate Comprehensive SEO Analysis</CardTitle>
                    <CardDescription className="text-gray-600">
                      Upload your Google Search Console export and get enterprise-grade insights powered by advanced analytics and AI
                    </CardDescription>
                  </CardHeader>
                  
                  <CardContent className="pt-6">
                    <form onSubmit={handleSubmit} className="space-y-6">
                      <div className="space-y-2">
                        <Label htmlFor="file" className="text-gray-700 font-medium">Search Console CSV Export</Label>
                        <FileUploadZone 
                          files={files}
                          onFilesChange={setFiles}
                        />
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
                          <div className="flex flex-col gap-2 text-xs text-gray-500 italic mt-3">
                            <div className="flex justify-between">
                              <span>Analyzing query patterns</span>
                              <span>{Math.min(100, progress * 1.8).toFixed(0)}%</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Researching keyword opportunities</span>
                              <span>{Math.min(100, progress * 1.7).toFixed(0)}%</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Identifying top competitors</span>
                              <span>{Math.min(100, progress * 1.5).toFixed(0)}%</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Generating AI recommendations</span>
                              <span>{Math.min(100, progress).toFixed(0)}%</span>
                            </div>
                          </div>
                        </div>
                      )}
                      
                      <Alert className="bg-purple-50 border-purple-200">
                        <Lightbulb className="h-4 w-4 text-purple-500" />
                        <AlertTitle className="text-purple-800 font-medium">Your SEMrush Alternative</AlertTitle>
                        <AlertDescription className="text-purple-700">
                          Get all the insights of premium SEO tools: keyword research, competitor analysis, content recommendations, 
                          and audience insights in a comprehensive report.
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
                          Generating advanced report...
                        </>
                      ) : (
                        <>Generate Enterprise-Grade SEO Report</>
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
              
              <TabsContent value="features">
                <Card className="border border-purple-100 shadow-lg">
                  <CardHeader className="bg-gradient-to-r from-purple-50 to-purple-100 rounded-t-lg">
                    <CardTitle className="text-2xl text-purple-800">Enterprise SEO Features</CardTitle>
                    <CardDescription className="text-gray-600">
                      Comprehensive tools to outperform your competition in search results
                    </CardDescription>
                  </CardHeader>
                  
                  <CardContent className="space-y-8 pt-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <Card className="border-purple-100">
                        <CardHeader className="bg-purple-50 rounded-t-lg border-b border-purple-100">
                          <div className="flex items-center gap-2">
                            <Search className="h-5 w-5 text-purple-600" />
                            <CardTitle className="text-lg text-purple-700">Keyword Research Engine</CardTitle>
                          </div>
                        </CardHeader>
                        <CardContent className="pt-4">
                          <ul className="space-y-3">
                            <li className="flex items-start gap-2">
                              <Badge className="mt-0.5 bg-purple-100 text-purple-700 hover:bg-purple-200">New</Badge>
                              <span className="text-sm">Automated keyword discovery for content gaps</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <Badge className="mt-0.5 bg-purple-100 text-purple-700 hover:bg-purple-200">New</Badge>
                              <span className="text-sm">Keyword difficulty analysis and ranking probability</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <Badge className="mt-0.5 bg-purple-100 text-purple-700 hover:bg-purple-200">New</Badge>
                              <span className="text-sm">Real-time search trend data integration</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <Badge className="mt-0.5 bg-purple-100 text-purple-700 hover:bg-purple-200">New</Badge>
                              <span className="text-sm">Related queries and semantic keyword clusters</span>
                            </li>
                          </ul>
                        </CardContent>
                      </Card>
                      
                      <Card className="border-purple-100">
                        <CardHeader className="bg-purple-50 rounded-t-lg border-b border-purple-100">
                          <div className="flex items-center gap-2">
                            <Users className="h-5 w-5 text-purple-600" />
                            <CardTitle className="text-lg text-purple-700">Competitor Intelligence</CardTitle>
                          </div>
                        </CardHeader>
                        <CardContent className="pt-4">
                          <ul className="space-y-3">
                            <li className="flex items-start gap-2">
                              <Badge className="mt-0.5 bg-purple-100 text-purple-700 hover:bg-purple-200">New</Badge>
                              <span className="text-sm">Automatic competitor identification from your niche</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <Badge className="mt-0.5 bg-purple-100 text-purple-700 hover:bg-purple-200">New</Badge>
                              <span className="text-sm">Competitor content analysis and keyword density</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <Badge className="mt-0.5 bg-purple-100 text-purple-700 hover:bg-purple-200">New</Badge>
                              <span className="text-sm">Ranking comparison for high-value keywords</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <Badge className="mt-0.5 bg-purple-100 text-purple-700 hover:bg-purple-200">New</Badge>
                              <span className="text-sm">Market share analysis and competitive positioning</span>
                            </li>
                          </ul>
                        </CardContent>
                      </Card>
                      
                      <Card className="border-purple-100">
                        <CardHeader className="bg-purple-50 rounded-t-lg border-b border-purple-100">
                          <div className="flex items-center gap-2">
                            <TrendingUp className="h-5 w-5 text-purple-600" />
                            <CardTitle className="text-lg text-purple-700">Advanced Analytics</CardTitle>
                          </div>
                        </CardHeader>
                        <CardContent className="pt-4">
                          <ul className="space-y-3">
                            <li className="flex items-start gap-2">
                              <Badge className="mt-0.5 bg-green-100 text-green-700 hover:bg-green-200">Enhanced</Badge>
                              <span className="text-sm">Statistical forecasting with trend prediction</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <Badge className="mt-0.5 bg-green-100 text-green-700 hover:bg-green-200">Enhanced</Badge>
                              <span className="text-sm">Query intent classification with NLP</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <Badge className="mt-0.5 bg-green-100 text-green-700 hover:bg-green-200">Enhanced</Badge>
                              <span className="text-sm">Seasonal pattern detection and opportunity alerts</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <Badge className="mt-0.5 bg-green-100 text-green-700 hover:bg-green-200">Enhanced</Badge>
                              <span className="text-sm">Multi-dimensional performance segmentation</span>
                            </li>
                          </ul>
                        </CardContent>
                      </Card>
                      
                      <Card className="border-purple-100">
                        <CardHeader className="bg-purple-50 rounded-t-lg border-b border-purple-100">
                          <div className="flex items-center gap-2">
                            <Microscope className="h-5 w-5 text-purple-600" />
                            <CardTitle className="text-lg text-purple-700">AI-Powered Strategy</CardTitle>
                          </div>
                        </CardHeader>
                        <CardContent className="pt-4">
                          <ul className="space-y-3">
                            <li className="flex items-start gap-2">
                              <Badge className="mt-0.5 bg-green-100 text-green-700 hover:bg-green-200">Enhanced</Badge>
                              <span className="text-sm">Content strategy recommendations powered by Gemini AI</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <Badge className="mt-0.5 bg-green-100 text-green-700 hover:bg-green-200">Enhanced</Badge>
                              <span className="text-sm">Prioritized action items with expected ROI estimates</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <Badge className="mt-0.5 bg-green-100 text-green-700 hover:bg-green-200">Enhanced</Badge>
                              <span className="text-sm">Channel-specific optimization suggestions</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <Badge className="mt-0.5 bg-green-100 text-green-700 hover:bg-green-200">Enhanced</Badge>
                              <span className="text-sm">Custom strategy tailored to your specific industry</span>
                            </li>
                          </ul>
                        </CardContent>
                      </Card>
                    </div>
                    
                    <Separator className="bg-purple-100" />
                    
                    <div className="space-y-4">
                      <h3 className="text-xl font-semibold text-purple-800">Comprehensive Analysis Modules</h3>
                      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                        <div className="bg-white p-4 rounded-lg border border-purple-100 shadow-sm">
                          <div className="flex items-center gap-2 mb-2">
                            <Network className="h-5 w-5 text-purple-600" />
                            <h4 className="font-medium text-purple-800">Query Analysis</h4>
                          </div>
                          <p className="text-xs text-gray-600">Discover patterns in search queries and identify high-potential keywords.</p>
                        </div>
                        
                        <div className="bg-white p-4 rounded-lg border border-purple-100 shadow-sm">
                          <div className="flex items-center gap-2 mb-2">
                            <Smartphone className="h-5 w-5 text-purple-600" />
                            <h4 className="font-medium text-purple-800">Device Insights</h4>
                          </div>
                          <p className="text-xs text-gray-600">Optimize for different devices based on performance metrics.</p>
                        </div>
                        
                        <div className="bg-white p-4 rounded-lg border border-purple-100 shadow-sm">
                          <div className="flex items-center gap-2 mb-2">
                            <Calendar className="h-5 w-5 text-purple-600" />
                            <h4 className="font-medium text-purple-800">Temporal Patterns</h4>
                          </div>
                          <p className="text-xs text-gray-600">Understand seasonal trends and forecast future performance.</p>
                        </div>
                        
                        <div className="bg-white p-4 rounded-lg border border-purple-100 shadow-sm">
                          <div className="flex items-center gap-2 mb-2">
                            <Globe2 className="h-5 w-5 text-purple-600" />
                            <h4 className="font-medium text-purple-800">Geographic Analysis</h4>
                          </div>
                          <p className="text-xs text-gray-600">Identify regional opportunities and optimize for local search.</p>
                        </div>
                        
                        <div className="bg-white p-4 rounded-lg border border-purple-100 shadow-sm">
                          <div className="flex items-center gap-2 mb-2">
                            <RadioTower className="h-5 w-5 text-purple-600" />
                            <h4 className="font-medium text-purple-800">Keyword Research</h4>
                          </div>
                          <p className="text-xs text-gray-600">Discover new keywords with trend analysis and difficulty scores.</p>
                        </div>
                        
                        <div className="bg-white p-4 rounded-lg border border-purple-100 shadow-sm">
                          <div className="flex items-center gap-2 mb-2">
                            <Users className="h-5 w-5 text-purple-600" />
                            <h4 className="font-medium text-purple-800">Competitor Analysis</h4>
                          </div>
                          <p className="text-xs text-gray-600">Benchmark against competitors and find content gaps.</p>
                        </div>
                        
                        <div className="bg-white p-4 rounded-lg border border-purple-100 shadow-sm">
                          <div className="flex items-center gap-2 mb-2">
                            <FileText className="h-5 w-5 text-purple-600" />
                            <h4 className="font-medium text-purple-800">Content Strategy</h4>
                          </div>
                          <p className="text-xs text-gray-600">Get AI-driven recommendations for content optimization.</p>
                        </div>
                        
                        <div className="bg-white p-4 rounded-lg border border-purple-100 shadow-sm">
                          <div className="flex items-center gap-2 mb-2">
                            <BarChart3 className="h-5 w-5 text-purple-600" />
                            <h4 className="font-medium text-purple-800">Advanced Visuals</h4>
                          </div>
                          <p className="text-xs text-gray-600">Interactive charts and dashboards for data-driven decisions.</p>
                        </div>
                      </div>
                    </div>
                    
                    <div className="bg-gradient-to-r from-purple-100 to-purple-50 p-6 rounded-xl shadow-sm">
                      <div className="flex items-center gap-3 mb-3">
                        <Trophy className="h-6 w-6 text-purple-700" />
                        <h3 className="text-xl font-semibold text-purple-800">SEMrush Alternative at Your Fingertips</h3>
                      </div>
                      <p className="text-gray-700 mb-4">
                        Get the power of premium SEO tools without the subscription cost. SEO Seer Pro delivers enterprise-grade 
                        insights for a fraction of the price, with complete data privacy and no recurring fees.
                      </p>
                      <Button 
                        onClick={navigateToUploadTab}
                        className="bg-gradient-to-r from-purple-700 to-purple-500 hover:from-purple-800 hover:to-purple-600"
                      >
                        Try SEO Seer Pro Now
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
              
              <TabsContent value="about">
                <Card className="border border-purple-100 shadow-lg">
                  <CardHeader className="bg-gradient-to-r from-purple-50 to-purple-100 rounded-t-lg">
                    <CardTitle className="text-2xl text-purple-800">About SEO Seer Pro</CardTitle>
                    <CardDescription className="text-gray-600">
                      Your comprehensive SEMrush alternative for data-driven SEO decisions
                    </CardDescription>
                  </CardHeader>
                  
                  <CardContent className="space-y-8 pt-6">
                    <div>
                      <h3 className="text-xl font-semibold mb-3 text-purple-800">What is SEO Seer Pro?</h3>
                      <p className="text-gray-600 leading-relaxed">
                        SEO Seer Pro is an advanced SEO analytics platform that transforms Google Search Console exports 
                        into comprehensive SEO reports with actionable insights. Designed as a cost-effective alternative to 
                        premium SEO tools like SEMrush, it provides keyword research, competitor analysis, content optimization 
                        recommendations, and AI-powered strategy suggestions in one powerful package.
                      </p>
                    </div>
                    
                    <Separator className="bg-purple-100" />
                    
                    <div className="space-y-6">
                      <h3 className="text-xl font-semibold text-purple-800">Core Capabilities</h3>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <FeatureCard 
                          title="Keyword Research" 
                          description="Discover new keywords with difficulty scores, trends, and related terms to expand your content strategy."
                          icon={RadioTower}
                        />
                        <FeatureCard 
                          title="Competitor Analysis" 
                          description="Identify top competitors, analyze their content strategy, and find opportunities to outrank them."
                          icon={Users}
                        />
                        <FeatureCard 
                          title="Content Optimization" 
                          description="Get AI-powered recommendations to improve existing content and create high-performing new pages."
                          icon={FileText}
                        />
                        <FeatureCard 
                          title="Performance Tracking" 
                          description="Track your SEO performance with comprehensive metrics, trend analysis, and forecasting."
                          icon={TrendingUp}
                        />
                        <FeatureCard 
                          title="International SEO" 
                          description="Geographic insights to optimize for global audiences and target regional opportunities."
                          icon={Globe2}
                        />
                        <FeatureCard 
                          title="Device Optimization" 
                          description="Ensure your site performs well across all devices with device-specific recommendations."
                          icon={Smartphone}
                        />
                      </div>
                    </div>
                    
                    <Separator className="bg-purple-100" />
                    
                    <div>
                      <h3 className="text-xl font-semibold mb-3 text-purple-800">How to Use</h3>
                      <ol className="list-decimal pl-5 space-y-3 text-gray-600">
                        <li>Export performance data from Google Search Console (supports both English and Spanish file names)</li>
                        <li>Upload any of these file types: Queries, Pages, Devices, Countries, Dates, Search Appearance, Filters</li>
                        <li>Provide your Gemini API key for AI-powered analysis</li>
                        <li>Click "Generate Enterprise-Grade SEO Report" and wait for processing</li>
                        <li>Download your comprehensive PDF report with visualization, data, and strategic recommendations</li>
                      </ol>
                    </div>
                    
                    <div className="bg-purple-50 p-6 rounded-xl border border-purple-100 flex flex-col sm:flex-row gap-6 items-center">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <FileText className="h-5 w-5 text-purple-600" />
                          <h3 className="text-lg font-semibold text-purple-800">Ready to try SEO Seer Pro?</h3>
                        </div>
                        <p className="text-gray-600 mb-4">
                          Upload your Google Search Console exports and discover actionable insights to improve your SEO performance today.
                        </p>
                        <Button 
                          onClick={navigateToUploadTab}
                          className="bg-gradient-to-r from-purple-700 to-purple-500 hover:from-purple-800 hover:to-purple-600"
                        >
                          Start Analyzing Now
                        </Button>
                      </div>
                      <div className="flex-1 flex justify-center">
                        <div className="p-4 bg-purple-100 rounded-xl">
                          <TrendingUp className="h-16 w-16 text-purple-700" />
                        </div>
                      </div>
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
