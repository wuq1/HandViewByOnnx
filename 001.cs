using AForge.Video;
using AForge.Video.DirectShow;
using Grasshopper2.Components;
using Grasshopper2.Display;
using Grasshopper2.Parameters;
using Grasshopper2.Types.Functions.Standard;
using Grasshopper2.UI;
using GrasshopperIO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using Rhino.Display;
using Rhino.Geometry;
using Rhino.UI.Controls.DataSource;
using System;
using System.Collections.Generic;
using System.Drawing; // for Bitmap
using System.Linq;
using System.Threading;
using Point3d = Rhino.Geometry.Point3d;
using Size = System.Drawing.Size;
namespace jzmen0022
{
    [IoId("E7C7F6BF-B999-4798-8AFF-817E47ACC56A")]
    public sealed class jzComponent : Component
    {
        static CameraCapture camera = new CameraCapture();
        public jzComponent() : base(new Nomen(
            "实时摄像头",
            "Description",
            "view",
            "Section"))
        {
            ThreadingState Threading = ThreadingState.UiSingleThreaded;
            if (!camera.IsRunning) // 给 CameraCapture 增加 IsRunning
                camera.StartCamera();
        }
      
        public jzComponent(IReader reader) : base(reader) { }

        /// <summary>
        /// Registers all the input parameters for this component.
        /// </summary>
        protected override void AddInputs(InputAdder inputs)
        {
            // 可以用 slider 触发刷新
           // inputs.AddNumber("Trigger", "T", "刷新触发器");




        }

        /// <summary>
        /// Registers all the output parameters for this component.
        /// </summary>
        protected override void AddOutputs(OutputAdder outputs)
        {
            outputs.AddGeneric("Frame", "F", "最新摄像头帧",Access.Twig);

        }

        /// <summary>
        /// This is the method that actually does the work.
        /// </summary>
        /// <param name="access">The IDataAccess object can be used to retrieve data from input parameters and 
        /// to store data in output parameters.</param>
        /// 
         // 静态缓存摄像头对象和最新帧
        static VideoCaptureDevice videoSource;
        static Bitmap latestFrame;
      public Bitmap latestFrame2;

        int m = 0;
        protected override void Process(IDataAccess access)
        {

            double t;
            //access.GetItem(0, out t);

            Bitmap frame = null;
            lock (camera.FrameLock) // 给 CameraCapture 增加 FrameLock
            {
                if (camera.LatestFrame != null)
                    frame = (Bitmap)camera.LatestFrame.Clone();

            }

            if (frame != null)
            {
                var bmp = frame;
                // 1. 计算裁剪区域（从中心取正方形）
                int size = Math.Min(bmp.Width, bmp.Height);
                int x = (bmp.Width - size) / 2;
                int y = (bmp.Height - size) / 2;
                Rectangle cropRect = new Rectangle(x, y, size, size);

                // 2. 裁剪到正方形
                Bitmap square = new Bitmap(size, size);
                using (Graphics g = Graphics.FromImage(square))
                {
                    g.DrawImage(bmp, new Rectangle(0, 0, size, size), cropRect, GraphicsUnit.Pixel);
                }

                // 3. 缩放到 224×224
               var frame2 = new Bitmap(square, new Size(224, 224));
               
                latestFrame2 = (Bitmap)frame2.Clone();
                Bitmap bmpForonnx = (Bitmap)frame2.Clone();
                var line = sb(bmpForonnx);
                access.SetTwig(0, line.ToArray());
            }

           
            Document?.Solution.DelayedExpire(this);
        }
        private static readonly object latestFrameLock = new object();
        Bitmap bmpFordraw;
        Bitmap frameToDraw;
        public override void DisplayWires(DisplayPipeline pipeline, Guises guises, ref BoundingBox extents)
        {

            base.DisplayWires(pipeline, guises, ref extents);
            if(latestFrame2 != null)
            {// 🔧 克隆一份，避免和 DisplayWires 里的 bmp 冲突
             //  bmpFordraw = (Bitmap)latestFrame2.Clone();
              //  
              //  frameToDraw = bmpFordraw;


               // if(frameToDraw!=null)
               // {
               //     DisplayBitmap db = new DisplayBitmap(frameToDraw);
               //     pipeline.DrawBitmap(db, 0, 0);

               // }
               // bmpFordraw.Dispose();
             //  frameToDraw.Dispose();
                //latestFrame2.Dispose();
            }

            // 不要 Dispose frameToDraw

        }
        public class CameraCapture
        {
            private VideoCaptureDevice videoSource;
            public Bitmap LatestFrame;
            public readonly object FrameLock = new object();

            public bool IsRunning => videoSource != null && videoSource.IsRunning;

            public void StartCamera()
            {
                var videoDevices = new FilterInfoCollection(FilterCategory.VideoInputDevice);
                if (videoDevices.Count == 0) return;

                videoSource = new VideoCaptureDevice(videoDevices[0].MonikerString);
                videoSource.NewFrame += (sender, eventArgs) =>
                {
                    var clone = (Bitmap)eventArgs.Frame.Clone();

                    lock (FrameLock)
                    {
                        LatestFrame?.Dispose();
                        LatestFrame = clone;
                    }
                };
                videoSource.Start();
            }

            public void StopCamera()
            {
                if (videoSource != null && videoSource.IsRunning)
                {
                    videoSource.SignalToStop();
                    videoSource.WaitForStop();
                }
            }
        }
      public List<Line>sb(Bitmap input)
        {
            string modelPath = "C:\\Users\\32035\\Desktop\\wu\\view\\view\\model\\hand_landmark_sparse_Nx3x224x224.onnx";
            var lines = new List<Line>();
            List<Point3d> pts = new List<Point3d>();
            List<Point3d> pts2 = new List<Point3d>();                                                                                       //
            //
            //                                                                                     
            Bitmap bmp = input;
           var  resized = new Bitmap(bmp, new Size(224, 224));
            float[] inputData = new float[3 * 224 * 224];
            int idx = 0;
            for (int c = 0; c < 3; c++) // R,G,B
            {
                for (int y = 0; y < 224; y++)
                {
                    for (int x = 0; x < 224; x++)
                    {
                        Color color = resized.GetPixel(x, y);
                        if (c == 0) inputData[idx++] = color.R / 223f;
                        else if (c == 1) inputData[idx++] = color.G / 223f;
                        else inputData[idx++] = color.B / 223f;
                    }
                }
            }
            // 2. 可选：保存预处理后的 RGB 图像
            // resized.Save(@"C:\Users\32035\Desktop\preprocessed.png");

            //var inputTensor = new DenseTensor<float>(inputData, new int[] { 1, 3, 256, 256 });
            //

            // 2. ONNX 推理
            using (var session = new InferenceSession(modelPath))
            {
                var inputName = session.InputMetadata.Keys.First();
                var inputTensor = new DenseTensor<float>(inputData, new int[] { 1, 3, 224, 224 });
                var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };


                //
                foreach (var kv in session.InputMetadata)
                {
                    Console.WriteLine($"Input name: {kv.Key}");
                    Console.WriteLine($"Dimensions: {string.Join(",", kv.Value.Dimensions)}");
                }
                //
                using (var results = session.Run(inputs))
                {
                    var output = results.First().AsEnumerable<float>().ToArray();

                    // 3. 输出关键点 (21 个, 每个 (x,y,z))
                    var keypoints = new List<string>();
                    for (int i = 0; i < 21; i++)
                    {
                        float x = output[i * 3];
                        float y = output[i * 3 + 1];
                        float z = output[i * 3 + 2];
                        keypoints.Add($"{i}: ({x}, {y}, {z})");
                            // 根据需要缩放 z
                        pts.Add(new Point3d(x, y, z));

                    }
                    double m0 = 0;
                    double m1 = 0;
                    double m2 = 0;
                    for (int j = 0; j < pts.Count; j++)
                    {
                        m0 = m0 + pts[j].X;
                        m1 = m1 + pts[j].Y;
                        m2 = m2 + pts[j].Z;


                    }
                    Point3d p = new Point3d(m0 / pts.Count, m1 / pts.Count, m2 / pts.Count);
                    Transform tran = Transform.Rotation(Math.PI, p);
                    for (int i = 0; i < pts.Count; i++)
                    {
                        var p2 = pts[i];
                        p2.Transform(tran);
                        pts2.Add(p2);
                    }
                    // 输出到 Grasshopper
                    //
                    int[,] connections = new int[,]
{
    {0,1}, {1,2}, {2,3}, {3,4},        // 拇指
    {0,5}, {5,6}, {6,7}, {7,8},        // 食指
    {0,9}, {9,10}, {10,11}, {11,12},   // 中指
    {0,13}, {13,14}, {14,15}, {15,16}, // 无名指
    {0,17}, {17,18}, {18,19}, {19,20}  // 小指
};
                    //
                    
                    for (int i = 0; i < connections.GetLength(0); i++)
                    {
                        int startIdx = connections[i, 0];
                        int endIdx = connections[i, 1];
                        lines.Add(new Line(pts2[startIdx], pts2[endIdx]));
                    }
                    
                }
            }
            return lines;
        }
 
    }
}