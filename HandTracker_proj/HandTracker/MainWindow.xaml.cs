using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Microsoft.Kinect;
using System.IO;
//using System.Drawing;
using MLApp;
using System.Threading;





namespace HandTracker
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private bool captureInProgress; // checks if capture is executing
        KinectSensor _sensor;
        private WriteableBitmap colorBitmap;
        CroppedBitmap cb;
        private WriteableBitmap aligned_colorBitmap;
       
        const int skeletonCount = 6;
        Skeleton[] allSkeletons = new Skeleton[skeletonCount];

        MLApp.MLApp matlab;
        private double[] greyHandImage; //
        int numberOfGestures = 5;
        private int[] gestures;
       

        private int old_pred = 0;
        private int pred; // Gives the final prediction to be sent to arduino.

        int x_cont = 0;

        double counter_gesture = 0;

        bool sendToSerial = false;
        
        

        public MainWindow()
        {
            InitializeComponent();
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {

            var activationContext = Type.GetTypeFromProgID("matlab.application.single");
            matlab = (MLApp.MLApp)Activator.CreateInstance(activationContext);

            gestures = new int[numberOfGestures];
            Console.WriteLine(matlab.Execute(@"cd D:\Thesis\Autoencoder5layers\"));
            //Console.WriteLine(matlab.Execute("imageData1 = []"));
            Console.WriteLine(matlab.Execute(@"load('D:\Thesis\Autoencoder5layers\PredictFinalFiles.mat')"));
           matlab.Execute("se = serial('COM4','BaudRate',115200);");
           matlab.Execute("fopen(se)");
            Console.WriteLine(matlab.Execute("pwd"));
            
            

        }

        private void btn_start_Click(object sender, RoutedEventArgs e)
        {
            
            if (KinectSensor.KinectSensors.Count > 0)
            {
                _sensor = KinectSensor.KinectSensors[0];
                if (_sensor.Status == KinectStatus.Connected)
                {

                    var parameters = new TransformSmoothParameters 
                    {
                        //Smoothing = 0.3f,
                        Smoothing = 0.1f,
                        Correction = 0.0f,
                        Prediction =0.1f,
                        JitterRadius =1.0f,
                        MaxDeviationRadius = 0.2f
                        

                    };

                    _sensor.ColorStream.Enable(ColorImageFormat.RgbResolution640x480Fps30);
                    _sensor.DepthStream.Enable(DepthImageFormat.Resolution640x480Fps30);
                   //_sensor.SkeletonStream.Enable();
                    _sensor.SkeletonStream.Enable(parameters);
                  
                    _sensor.Start();
                    _sensor.ElevationAngle = -17; // Elevation angle of kinect

                    //*****************************

                    if (captureInProgress)
                    {  //if camera is getting frames then stop the capture and set button Text
                        // "Start" for resuming capture
                        btn_start.Content = "Start!"; //

                        _sensor.AllFramesReady -= _sensor_AllFramesReady;
                    }
                    else
                    {
                        //if camera is NOT getting frames then start the capture and set button
                        // Text to "Stop" for pausing capture
                        btn_start.Content = "Stop";

                        _sensor.AllFramesReady += _sensor_AllFramesReady;
                    }

                    captureInProgress = !captureInProgress;

                    //*****************************

                  

                }
            }
        }

        void _sensor_AllFramesReady(object sender, AllFramesReadyEventArgs e)
        {
            using (ColorImageFrame colorFrame = e.OpenColorImageFrame())
            {
                using (DepthImageFrame depthFrame = e.OpenDepthImageFrame())
                {
                    using (SkeletonFrame skeletonFrame = e.OpenSkeletonFrame())
                       {
                        if ( colorFrame == null || depthFrame == null|| skeletonFrame == null)
                        {
                            return;
                        }

                        // ****************Color Frame***************
                        ColorFrameToBitmap(colorFrame);
                        //img_color.Source = colorBitmap;



                        //************Skeleton Frame*******************

                        Skeleton user_skeleton = getSkeleton(skeletonFrame);

                        if (user_skeleton == null)
                            return;
                        //********From here --->>> Get the hand pixels
                        // void get hand coordinates
                        GetROI(user_skeleton,depthFrame,colorFrame);


                       //img_color.Source = aligned_colorBitmap;
                        img_color.Source = colorBitmap;
                       img_cropped.Source = cb;

                      

                        

                    }
            }
        }
    }

        void GetROI(Skeleton user, DepthImageFrame depthFrame , ColorImageFrame color_frame = null)
        {

            
            // Map skeleton to Depth 
            DepthImagePoint rightHandPoint = 
                _sensor.CoordinateMapper.MapSkeletonPointToDepthPoint(user.Joints[JointType.HandRight].Position, DepthImageFormat.Resolution640x480Fps30);

            DepthImagePoint rightWristPoint =
                _sensor.CoordinateMapper.MapSkeletonPointToDepthPoint(user.Joints[JointType.WristRight].Position, DepthImageFormat.Resolution640x480Fps30);

            int hand_depth = (rightHandPoint.Depth>rightWristPoint.Depth)?rightHandPoint.Depth:rightWristPoint.Depth+10; // hand depth used for segmenting out the hand


            //*********************************** Map The depth Image to color Image to align the color image************************************************************************

            DepthImagePixel[] depthImagePixels = new DepthImagePixel[depthFrame.PixelDataLength];
            depthFrame.CopyDepthImagePixelDataTo(depthImagePixels);

            short[] rawDepthData = new short[depthFrame.PixelDataLength];
            depthFrame.CopyPixelDataTo(rawDepthData);

            ColorImagePoint[] mapped_depth_locations = new ColorImagePoint[depthFrame.PixelDataLength];

            _sensor.CoordinateMapper.MapDepthFrameToColorFrame(DepthImageFormat.Resolution640x480Fps30, depthImagePixels, ColorImageFormat.RgbResolution640x480Fps30, mapped_depth_locations);
            byte[] aligned_colorPixels = new byte[color_frame.PixelDataLength];  // creating a byte array for storing the aligned pixel values

            byte[] original_colorPixels = new byte[color_frame.PixelDataLength];
            color_frame.CopyPixelDataTo(original_colorPixels);
            int aligned_image_index = 0;
            //int hand_baseindex = rightHandPoint.Y*640 + rightHandPoint.X;
            for (int i = 0; i < mapped_depth_locations.Length; i++)
            {
                
                int depth = rawDepthData[i] >> DepthImageFrame.PlayerIndexBitmaskWidth;
                //Console.WriteLine(depth);
                ColorImagePoint point = mapped_depth_locations[i];
                
                if ((point.X >= 0 && point.X < 640) && (point.Y >= 0 && point.Y < 480))
                {
                    int baseIndex = (point.Y * 640 + point.X) * 4;
                    if (depth < hand_depth && depth != -1)
                    {
                        
                        aligned_colorPixels[aligned_image_index] = original_colorPixels[baseIndex];
                        aligned_colorPixels[aligned_image_index + 1] = original_colorPixels[baseIndex + 1];
                        aligned_colorPixels[aligned_image_index + 2] = original_colorPixels[baseIndex + 2];
                        aligned_colorPixels[aligned_image_index + 3] = 0;
                    }
                    else
                    {
                        aligned_colorPixels[aligned_image_index] = 0;
                        aligned_colorPixels[aligned_image_index + 1] = 0;
                        aligned_colorPixels[aligned_image_index + 2] = 0;
                        aligned_colorPixels[aligned_image_index + 3] = 0;
                    }
                } 
                aligned_image_index = aligned_image_index + 4;
               


                // *************************** Now modify the contents of this aligned_colorBitmap using the depth information ***************************************************
              


            }


            //***********************************************************************************************************************************************************************



            


            int threshold = 20;
            
            int hand_length = 3 * Math.Max(Math.Abs(rightHandPoint.X - rightWristPoint.X), Math.Abs(rightHandPoint.Y - rightWristPoint.Y));

          //  int hand_length = (int)Math.Sqrt((rightHandPoint.X - rightWristPoint.X) ^ 2 + (rightHandPoint.Y - rightWristPoint.Y) ^ 2);
           
            int hand_length_old = hand_length;
            //****************************Low pass filter for hand_length*********************************

            if (Math.Abs(hand_length - hand_length_old) > threshold)
                hand_length = hand_length_old;

            //************************************************************************************************

           // Console.WriteLine(hand_length);
            int top_left_X_depth = rightHandPoint.X - hand_length;
            int top_left_Y_depth = rightHandPoint.Y - hand_length;
            int top_left_Z_depth = rightHandPoint.Depth;


            top_left_X_depth = (top_left_X_depth<0)? 0 : top_left_X_depth;
            top_left_Y_depth = (top_left_Y_depth<0)? 0 : top_left_Y_depth;

            DepthImagePoint top_left = new DepthImagePoint();
            top_left.X = top_left_X_depth;
            top_left.Y = top_left_Y_depth;
            top_left.Depth = rightHandPoint.Depth;

            int bottom_right_X_depth = rightHandPoint.X + hand_length;
            int bottom_right_Y_depth = rightHandPoint.Y + hand_length;
            int bottom_right_Z_depth = rightHandPoint.Depth ;

            bottom_right_X_depth = (bottom_right_X_depth>640)? 600 : bottom_right_X_depth;
            bottom_right_Y_depth = (bottom_right_Y_depth>480)? 400 : bottom_right_Y_depth;

            DepthImagePoint bottom_right = new DepthImagePoint();
            bottom_right.X = bottom_right_X_depth;
            bottom_right.Y = bottom_right_Y_depth;
            bottom_right.Depth = bottom_right_Z_depth;

            Canvas.SetLeft(right_hand_pointer, top_left.X - right_hand_pointer.Width / 2);
            Canvas.SetTop(right_hand_pointer, top_left.Y - right_hand_pointer.Height / 2);

           


            Canvas.SetLeft(left_hand_pointer, bottom_right.X - left_hand_pointer.Width / 2);
            Canvas.SetTop(left_hand_pointer, bottom_right.Y - left_hand_pointer.Height / 2);

          border_rect.Width = 2*hand_length;
          border_rect.Height = 2*hand_length;

          Canvas.SetLeft(border_rect, top_left.X);
          Canvas.SetTop(border_rect, top_left.Y);


          aligned_colorPixelsToBitmap(aligned_colorPixels, color_frame, (int)top_left.X, (int)top_left.Y, (int)border_rect.Width, (int)border_rect.Height);   

           
        
        }

       
        

        void ColorFrameToBitmap(ColorImageFrame colorFrame, int x=0, int y=0, int width=0, int height=0)
        {

            byte[] color_pixels = new byte[colorFrame.PixelDataLength];
            colorFrame.CopyPixelDataTo(color_pixels);
            int stride = colorFrame.Width * 4;
            if(width==0 && height==0)
                colorBitmap = new WriteableBitmap(_sensor.ColorStream.FrameWidth,_sensor.ColorStream.FrameHeight,96,96,PixelFormats.Bgr32,null);
            else
                colorBitmap = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgr32, null);
                    
                         // Write the pixel data into our bitmap
                         this.colorBitmap.WritePixels(
                           new Int32Rect(x, y, this.colorBitmap.PixelWidth, this.colorBitmap.PixelHeight),
                           color_pixels,
                           this.colorBitmap.PixelWidth * sizeof(int),
                           0);
            
         }


        
        void aligned_colorPixelsToBitmap(byte[] pixels, ColorImageFrame colorFrame, int x = 0, int y = 0, int width = 0, int height = 0)
        {
            
            width = width < 95 ? 95 : width;
            height = height < 95 ? 95 : height;
            int stride = colorFrame.Width * 4;
            aligned_colorBitmap = new WriteableBitmap(_sensor.ColorStream.FrameWidth, _sensor.ColorStream.FrameHeight, 96, 96, PixelFormats.Bgr32, null);
            
            
            
            
            // Write the pixel data into our bitmap
            this.aligned_colorBitmap.WritePixels(
              new Int32Rect(0, 0, this.aligned_colorBitmap.PixelWidth, this.aligned_colorBitmap.PixelHeight),
              pixels,
              this.aligned_colorBitmap.PixelWidth * sizeof(int),
              0);

           
            if (width > 0 && height > 0)
            {
                x = (x - 20) > 0 ? x - 20 : 0;
                y = (y - 20) > 0 ? y - 20 : 0;
                

                if (x + width >= 640 || y + height >= 480)
                    return;

                if (width > 0 && height > 0)
                {
                    cb = new CroppedBitmap(aligned_colorBitmap, new Int32Rect(x, y, width, height));

                    //************************ Convert Cropped image to double array of pixel intensities and send to matlab************************** 


                    int array_len = cb.PixelHeight * cb.PixelWidth * 4;
                    byte[] px = new byte[array_len];
                    int stride_cb = cb.PixelWidth * 4;

                    cb.CopyPixels(px, stride_cb, 0);

                    greyHandImage = new double[array_len / 4];

                    // Convert RGBA to Greyscale double

                    for (int i = 0, greyIndex = 0; i < array_len - 4; greyIndex++)
                    {
                        double Red = px[i];
                        double Green = px[i + 1];
                        double Blue = px[i + 2];
                        i = i + 4;


                        greyHandImage[greyIndex] = (Red * 0.212600 + Green * 0.715200 + Blue * 0.072200) /* /255 */;
                       // Console.WriteLine(greyHandImage[greyIndex]);
                    }

                  
                    // Write to matlab
                   
                    matlab.PutWorkspaceData("im", "base", greyHandImage);
                    

                    matlab.Execute("row = floor(sqrt(length(im)));");
                    matlab.Execute("im_1 = reshape(im ,row,row);");
                    matlab.Execute("im_2 = imresize(im_1' , [28 28]);");
                    matlab.Execute("testdata = im_2(:);");
                    matlab.Execute("testdata = (testdata-avg)./sd;");

                   
                      
                        matlab.Execute("[out_class] = stackedAEPredict(stackedAEOptTheta, length(testdata), 200, 5, netconfig, testdata)");
                        
                        int out_class = Convert.ToInt32( matlab.GetVariable("out_class", "base"));


                        if (x_cont == out_class)
                        {
                            counter_gesture++;

                            if (counter_gesture == 13) // 10 to 15 frames checking gives better result
                            {
                                pred = x_cont;
                                counter_gesture = 1;
                            }
                            
                        }
                        else
                        {
                            counter_gesture = 0;
                            x_cont = out_class;
                        }

                
                        if (old_pred != pred)
                        {
                            if (pred == 5)
                            {
                                matlab.Execute("fprintf(se,'%c','5')");
                                sendToSerial = false;
                            }
                            else if (pred == 1 || pred ==2 || pred ==4)
                                sendToSerial = true;
                           
                            

                            if (sendToSerial)
                            {
                                 matlab.PutWorkspaceData("predToSerial", "base", pred);
                                 matlab.Execute("predToSerial = num2str(predToSerial);");
                                 matlab.Execute("fprintf(se,'%c',predToSerial)");
                                // Console.WriteLine(matlab.Execute("imshow((im_1./255)');"));
                               
 
                            }
                           
                            old_pred = pred;
                            Console.WriteLine(pred);
                            
                        }



                       /* To Save image as file
                        * Console.WriteLine(matlab.Execute("imageData1 = [imageData1 testdata];"));
                        matlab.PutWorkspaceData("counter_matlab", "base", counter_matlab);
                        Console.WriteLine(matlab.Execute(@"img_path = sprintf('%s%d%s','C:\Users\Varun Kumar\Dropbox\Codes\Thesis\Data\4_back\',counter_matlab,'.jpeg');"));
                        Console.WriteLine(matlab.Execute("imwrite(im_1', img_path);"));
                        Console.WriteLine(matlab.Execute("imshow((im_1./255)');"));
                        Console.WriteLine("%s %d", "Sent data", counter_matlab);
                        counter_gesture++;
                        */
                }

               
//******************************************************************************************************************
                    img_cropped.Source = aligned_colorBitmap;
                    
               
            }
           
        }

       
        Skeleton getSkeleton(SkeletonFrame skeletonFrame)
        {
            skeletonFrame.CopySkeletonDataTo(allSkeletons);
            

            Skeleton user_skeleton = null;
            float depth = float.MaxValue;
            foreach (Skeleton sk in allSkeletons)
            {
                if (sk.TrackingState == SkeletonTrackingState.Tracked)
                {
                    if (sk.Joints[JointType.HipCenter].Position.Z < depth)
                    {
                        // Console.WriteLine(depth);
                        user_skeleton = sk;
                        depth = sk.Joints[JointType.HipCenter].Position.Z;
                    }
                }
            }

            return user_skeleton;
        }
        
        
       

        void stopKinect(KinectSensor sensor)
        {
            if (sensor != null)
            {
                sensor.Stop();
                sensor.AudioSource.Stop();
                matlab.Execute("fclose(se)");
            }
        }

       
    }
     
}
//
//Interfacing C# with matlab http://www.codeproject.com/Articles/594636/Using-Matlab-from-a-Csharp-application

