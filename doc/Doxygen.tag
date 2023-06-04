<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<tagfile doxygen_version="1.9.1">
  <compound kind="file">
    <name>Device_PyBind_Module.cpp</name>
    <path>/mnt/c/Users/Osman/Desktop/Projects/Public Repos/PySYCL/src/Device/</path>
    <filename>_device___py_bind___module_8cpp.html</filename>
    <includes id="_s_y_c_l___device_8h" name="SYCL_Device.h" local="yes" imported="no">SYCL_Device.h</includes>
    <includes id="_s_y_c_l___device___inquiry_8h" name="SYCL_Device_Inquiry.h" local="yes" imported="no">SYCL_Device_Inquiry.h</includes>
  </compound>
  <compound kind="file">
    <name>SYCL_Device.h</name>
    <path>/mnt/c/Users/Osman/Desktop/Projects/Public Repos/PySYCL/src/Device/</path>
    <filename>_s_y_c_l___device_8h.html</filename>
    <class kind="class">pysycl::SYCL_Device</class>
    <namespace>pysycl</namespace>
  </compound>
  <compound kind="file">
    <name>SYCL_Device_Inquiry.h</name>
    <path>/mnt/c/Users/Osman/Desktop/Projects/Public Repos/PySYCL/src/Device/</path>
    <filename>_s_y_c_l___device___inquiry_8h.html</filename>
    <namespace>pysycl</namespace>
    <member kind="function">
      <type>std::vector&lt; std::string &gt;</type>
      <name>platform_list</name>
      <anchorfile>namespacepysycl.html</anchorfile>
      <anchor>a87f73272c1b9023c7b0cd574d527ffd3</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; std::string &gt;</type>
      <name>device_list</name>
      <anchorfile>namespacepysycl.html</anchorfile>
      <anchor>aa743f37417afb11704024adba84a3493</anchor>
      <arglist>(int platform_index)</arglist>
    </member>
    <member kind="function">
      <type>sycl::queue</type>
      <name>get_queue</name>
      <anchorfile>namespacepysycl.html</anchorfile>
      <anchor>a00b26210f29a913df10ccb3b63c13fb5</anchor>
      <arglist>(int platform_index=0, int device_index=0)</arglist>
    </member>
  </compound>
  <compound kind="file">
    <name>Math_Functions.h</name>
    <path>/mnt/c/Users/Osman/Desktop/Projects/Public Repos/PySYCL/src/Math/</path>
    <filename>_math___functions_8h.html</filename>
    <namespace>pysycl</namespace>
    <member kind="function">
      <type>auto</type>
      <name>add_function</name>
      <anchorfile>namespacepysycl.html</anchorfile>
      <anchor>ae5f66ba2d2ddc522cb092b1fca015fa5</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>subtract_function</name>
      <anchorfile>namespacepysycl.html</anchorfile>
      <anchor>aeed90e9d7c12f5cb915eb8d0fe5586b3</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>multiply_function</name>
      <anchorfile>namespacepysycl.html</anchorfile>
      <anchor>ab54eaad7760b2d4a262d702a22cca24c</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>divide_function</name>
      <anchorfile>namespacepysycl.html</anchorfile>
      <anchor>aa4e82e40c9ee3367cf4ce417cf928119</anchor>
      <arglist>()</arglist>
    </member>
  </compound>
  <compound kind="file">
    <name>Vector_Object.h</name>
    <path>/mnt/c/Users/Osman/Desktop/Projects/Public Repos/PySYCL/src/Vector/</path>
    <filename>_vector___object_8h.html</filename>
    <includes id="_s_y_c_l___device___inquiry_8h" name="SYCL_Device_Inquiry.h" local="yes" imported="no">../Device/SYCL_Device_Inquiry.h</includes>
    <includes id="_math___functions_8h" name="Math_Functions.h" local="yes" imported="no">../Math/Math_Functions.h</includes>
    <class kind="class">pysycl::Vector_Object</class>
    <namespace>pysycl</namespace>
  </compound>
  <compound kind="file">
    <name>Vector_Operations.h</name>
    <path>/mnt/c/Users/Osman/Desktop/Projects/Public Repos/PySYCL/src/Vector/</path>
    <filename>_vector___operations_8h.html</filename>
    <includes id="_s_y_c_l___device___inquiry_8h" name="SYCL_Device_Inquiry.h" local="yes" imported="no">../Device/SYCL_Device_Inquiry.h</includes>
    <includes id="_math___functions_8h" name="Math_Functions.h" local="yes" imported="no">../Math/Math_Functions.h</includes>
    <namespace>pysycl</namespace>
    <member kind="function">
      <type>std::vector&lt; double &gt;</type>
      <name>Vector_Operation</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>ga5b23a7f925405b53f39962c2ce4c1501</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, std::vector&lt; double &gt; vector_b, sycl::queue Q, Function_T function)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; double &gt;</type>
      <name>Vector_Addition</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>ga7bfd0f0fdf9ddae44956c1b9449f8943</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, std::vector&lt; double &gt; vector_b, int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; double &gt;</type>
      <name>Vector_Subtraction</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>ga49293a8948584a3f1e7f60c44f85a27c</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, std::vector&lt; double &gt; vector_b, int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; double &gt;</type>
      <name>Vector_Element_Multiplication</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>gafe212cc17352d0535341ef351a818456</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, std::vector&lt; double &gt; vector_b, int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; double &gt;</type>
      <name>Vector_Element_Division</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>gadc8ccf79f5be15b1ed782efd21b1bd5f</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, std::vector&lt; double &gt; vector_b, int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>Vector_Sum_Reduction</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>gad654c20d23fa2c428619536f6f850651</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>Vector_Min_Reduction</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>ga363773f976560ac01dbbc23baa98ccc0</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>Vector_Max_Reduction</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>ga31e509922e8e5cc38cbd36a38f6e96fe</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>Vector_Dot_Product</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>gad27fb35fc715e166f6a11336cf293b8a</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, std::vector&lt; double &gt; vector_b, int platform_index=0, int device_index=0)</arglist>
    </member>
  </compound>
  <compound kind="file">
    <name>Vector_PyBind_Module.cpp</name>
    <path>/mnt/c/Users/Osman/Desktop/Projects/Public Repos/PySYCL/src/Vector/</path>
    <filename>_vector___py_bind___module_8cpp.html</filename>
    <includes id="_vector___operations_8h" name="Vector_Operations.h" local="yes" imported="no">Vector_Operations.h</includes>
    <includes id="_vector___object_8h" name="Vector_Object.h" local="yes" imported="no">Vector_Object.h</includes>
  </compound>
  <compound kind="class">
    <name>pysycl::SYCL_Device</name>
    <filename>classpysycl_1_1_s_y_c_l___device.html</filename>
    <member kind="function">
      <type></type>
      <name>SYCL_Device</name>
      <anchorfile>classpysycl_1_1_s_y_c_l___device.html</anchorfile>
      <anchor>ab8bd2aa65649a5e947107f485b844560</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>SYCL_Device</name>
      <anchorfile>classpysycl_1_1_s_y_c_l___device.html</anchorfile>
      <anchor>a7dc4fb855558cabaf71de2740b51e65b</anchor>
      <arglist>(const SYCL_Device &amp;)=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>SYCL_Device</name>
      <anchorfile>classpysycl_1_1_s_y_c_l___device.html</anchorfile>
      <anchor>a8409f90c13250f8801d94dcc01687756</anchor>
      <arglist>(SYCL_Device &amp;&amp;)=default</arglist>
    </member>
    <member kind="function">
      <type>SYCL_Device &amp;</type>
      <name>operator=</name>
      <anchorfile>classpysycl_1_1_s_y_c_l___device.html</anchorfile>
      <anchor>af82bd5395f0e24297f9ecb8f09753c88</anchor>
      <arglist>(const SYCL_Device &amp;)=default</arglist>
    </member>
    <member kind="function">
      <type>SYCL_Device &amp;</type>
      <name>operator=</name>
      <anchorfile>classpysycl_1_1_s_y_c_l___device.html</anchorfile>
      <anchor>a1f26d5b2897a6a1973fcfac438250165</anchor>
      <arglist>(SYCL_Device &amp;&amp;)=default</arglist>
    </member>
    <member kind="function">
      <type>std::string</type>
      <name>device_name</name>
      <anchorfile>classpysycl_1_1_s_y_c_l___device.html</anchorfile>
      <anchor>a615ce057440ab7e14ebad8d0f59d68f8</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>std::string</type>
      <name>device_vendor</name>
      <anchorfile>classpysycl_1_1_s_y_c_l___device.html</anchorfile>
      <anchor>a9c95ff44cda22de7830f413e5c84e12e</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>SYCL_Device</name>
      <anchorfile>classpysycl_1_1_s_y_c_l___device.html</anchorfile>
      <anchor>a5695f1876c8f6cd36670da635f3790a3</anchor>
      <arglist>(int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="variable" protection="private">
      <type>sycl::queue</type>
      <name>sycl_device</name>
      <anchorfile>classpysycl_1_1_s_y_c_l___device.html</anchorfile>
      <anchor>a77baa00848741a23ebd5151fd97b5927</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>pysycl::Vector_Object</name>
    <filename>classpysycl_1_1_vector___object.html</filename>
    <member kind="function">
      <type></type>
      <name>Vector_Object</name>
      <anchorfile>classpysycl_1_1_vector___object.html</anchorfile>
      <anchor>ae03d87fb6810afb959f7c9272a0a789f</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Vector_Object</name>
      <anchorfile>classpysycl_1_1_vector___object.html</anchorfile>
      <anchor>abd2c93a17fad4b6b15c97aa156a28bb9</anchor>
      <arglist>(const Vector_Object &amp;)=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Vector_Object</name>
      <anchorfile>classpysycl_1_1_vector___object.html</anchorfile>
      <anchor>a728193ccda126b7c809746ef5d03fbd7</anchor>
      <arglist>(Vector_Object &amp;&amp;)=default</arglist>
    </member>
    <member kind="function">
      <type>Vector_Object &amp;</type>
      <name>operator=</name>
      <anchorfile>classpysycl_1_1_vector___object.html</anchorfile>
      <anchor>a0ba652ae3f01353ea6adaa62ece40b27</anchor>
      <arglist>(const Vector_Object &amp;)=default</arglist>
    </member>
    <member kind="function">
      <type>Vector_Object &amp;</type>
      <name>operator=</name>
      <anchorfile>classpysycl_1_1_vector___object.html</anchorfile>
      <anchor>a1be42a64dabbe19dfc6d164ade1fe2f9</anchor>
      <arglist>(Vector_Object &amp;&amp;)=default</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_data</name>
      <anchorfile>classpysycl_1_1_vector___object.html</anchorfile>
      <anchor>a217bbe42ce68ce68f1221d34f83ea18d</anchor>
      <arglist>(std::vector&lt; double &gt; data_in)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>reset_data</name>
      <anchorfile>classpysycl_1_1_vector___object.html</anchorfile>
      <anchor>a7db68fb66bd2af3924ee56cb30a0693b</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; double &gt;</type>
      <name>get_data</name>
      <anchorfile>classpysycl_1_1_vector___object.html</anchorfile>
      <anchor>a20cb73c4a8faad0182698e63fccdaa4a</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>element_vector_operation</name>
      <anchorfile>classpysycl_1_1_vector___object.html</anchorfile>
      <anchor>af8bf96878ce9dc0c685e93378d5b5a8f</anchor>
      <arglist>(Function_type function, std::vector&lt; double &gt; data_in)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>element_vector_operation</name>
      <anchorfile>classpysycl_1_1_vector___object.html</anchorfile>
      <anchor>ab300bca4c6600f4a296321caf9a11457</anchor>
      <arglist>(Function_type function, double C)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>add_vector</name>
      <anchorfile>classpysycl_1_1_vector___object.html</anchorfile>
      <anchor>ae3da95a626954cfbcf1463d9be19d7b2</anchor>
      <arglist>(std::vector&lt; double &gt; vector_in)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>add_constant</name>
      <anchorfile>classpysycl_1_1_vector___object.html</anchorfile>
      <anchor>aeb15f2d16212072d5c9d6f2861480abf</anchor>
      <arglist>(double C)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>subtract_vector</name>
      <anchorfile>classpysycl_1_1_vector___object.html</anchorfile>
      <anchor>ae85b4af453157f0b3a1f2b3b77a4bc03</anchor>
      <arglist>(std::vector&lt; double &gt; vector_in)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>subtract_constant</name>
      <anchorfile>classpysycl_1_1_vector___object.html</anchorfile>
      <anchor>adc000c855fc94eded62c351ac8e1a2c1</anchor>
      <arglist>(double C)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>multiply_vector</name>
      <anchorfile>classpysycl_1_1_vector___object.html</anchorfile>
      <anchor>ad53ed051526f76e70b93d6d2db7bf096</anchor>
      <arglist>(std::vector&lt; double &gt; vector_in)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>multiply_constant</name>
      <anchorfile>classpysycl_1_1_vector___object.html</anchorfile>
      <anchor>ad5398f3e58ff2b046da19346de06bb7b</anchor>
      <arglist>(double C)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>divide_vector</name>
      <anchorfile>classpysycl_1_1_vector___object.html</anchorfile>
      <anchor>a39582ffa9362c69728aa78f6306cc1fb</anchor>
      <arglist>(std::vector&lt; double &gt; vector_in)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>divide_constant</name>
      <anchorfile>classpysycl_1_1_vector___object.html</anchorfile>
      <anchor>a0101115bf2e234f0d8538c5af0fd2e61</anchor>
      <arglist>(double C)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Vector_Object</name>
      <anchorfile>classpysycl_1_1_vector___object.html</anchorfile>
      <anchor>a626649b50aac046ba0c7b977e1e8e69a</anchor>
      <arglist>(size_t N, int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="variable" protection="private">
      <type>size_t</type>
      <name>N</name>
      <anchorfile>classpysycl_1_1_vector___object.html</anchorfile>
      <anchor>a522c6fb0bfd2b7b70e57884e19226257</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="private">
      <type>double *</type>
      <name>data_device_in</name>
      <anchorfile>classpysycl_1_1_vector___object.html</anchorfile>
      <anchor>ab75e1dfd536dcdf738322c1e597bde3a</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="private">
      <type>double *</type>
      <name>data_device</name>
      <anchorfile>classpysycl_1_1_vector___object.html</anchorfile>
      <anchor>a72b082c7ed9860a5a366b2b726ff03c4</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="private">
      <type>sycl::queue</type>
      <name>device_queue</name>
      <anchorfile>classpysycl_1_1_vector___object.html</anchorfile>
      <anchor>ab156ed26efbafdac904c2c01f55950b1</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>pysycl</name>
    <filename>namespacepysycl.html</filename>
    <class kind="class">pysycl::SYCL_Device</class>
    <class kind="class">pysycl::Vector_Object</class>
    <member kind="function">
      <type>std::vector&lt; std::string &gt;</type>
      <name>platform_list</name>
      <anchorfile>namespacepysycl.html</anchorfile>
      <anchor>a87f73272c1b9023c7b0cd574d527ffd3</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; std::string &gt;</type>
      <name>device_list</name>
      <anchorfile>namespacepysycl.html</anchorfile>
      <anchor>aa743f37417afb11704024adba84a3493</anchor>
      <arglist>(int platform_index)</arglist>
    </member>
    <member kind="function">
      <type>sycl::queue</type>
      <name>get_queue</name>
      <anchorfile>namespacepysycl.html</anchorfile>
      <anchor>a00b26210f29a913df10ccb3b63c13fb5</anchor>
      <arglist>(int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>add_function</name>
      <anchorfile>namespacepysycl.html</anchorfile>
      <anchor>ae5f66ba2d2ddc522cb092b1fca015fa5</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>subtract_function</name>
      <anchorfile>namespacepysycl.html</anchorfile>
      <anchor>aeed90e9d7c12f5cb915eb8d0fe5586b3</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>multiply_function</name>
      <anchorfile>namespacepysycl.html</anchorfile>
      <anchor>ab54eaad7760b2d4a262d702a22cca24c</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>divide_function</name>
      <anchorfile>namespacepysycl.html</anchorfile>
      <anchor>aa4e82e40c9ee3367cf4ce417cf928119</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; double &gt;</type>
      <name>Vector_Operation</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>ga5b23a7f925405b53f39962c2ce4c1501</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, std::vector&lt; double &gt; vector_b, sycl::queue Q, Function_T function)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; double &gt;</type>
      <name>Vector_Addition</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>ga7bfd0f0fdf9ddae44956c1b9449f8943</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, std::vector&lt; double &gt; vector_b, int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; double &gt;</type>
      <name>Vector_Subtraction</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>ga49293a8948584a3f1e7f60c44f85a27c</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, std::vector&lt; double &gt; vector_b, int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; double &gt;</type>
      <name>Vector_Element_Multiplication</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>gafe212cc17352d0535341ef351a818456</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, std::vector&lt; double &gt; vector_b, int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; double &gt;</type>
      <name>Vector_Element_Division</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>gadc8ccf79f5be15b1ed782efd21b1bd5f</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, std::vector&lt; double &gt; vector_b, int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>Vector_Sum_Reduction</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>gad654c20d23fa2c428619536f6f850651</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>Vector_Min_Reduction</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>ga363773f976560ac01dbbc23baa98ccc0</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>Vector_Max_Reduction</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>ga31e509922e8e5cc38cbd36a38f6e96fe</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>Vector_Dot_Product</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>gad27fb35fc715e166f6a11336cf293b8a</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, std::vector&lt; double &gt; vector_b, int platform_index=0, int device_index=0)</arglist>
    </member>
  </compound>
  <compound kind="group">
    <name>Device</name>
    <title>Device</title>
    <filename>group___device.html</filename>
  </compound>
  <compound kind="group">
    <name>Math</name>
    <title>Math</title>
    <filename>group___math.html</filename>
  </compound>
  <compound kind="group">
    <name>Vector</name>
    <title>Vector</title>
    <filename>group___vector.html</filename>
    <member kind="function">
      <type>std::vector&lt; double &gt;</type>
      <name>Vector_Operation</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>ga5b23a7f925405b53f39962c2ce4c1501</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, std::vector&lt; double &gt; vector_b, sycl::queue Q, Function_T function)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; double &gt;</type>
      <name>Vector_Addition</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>ga7bfd0f0fdf9ddae44956c1b9449f8943</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, std::vector&lt; double &gt; vector_b, int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; double &gt;</type>
      <name>Vector_Subtraction</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>ga49293a8948584a3f1e7f60c44f85a27c</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, std::vector&lt; double &gt; vector_b, int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; double &gt;</type>
      <name>Vector_Element_Multiplication</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>gafe212cc17352d0535341ef351a818456</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, std::vector&lt; double &gt; vector_b, int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; double &gt;</type>
      <name>Vector_Element_Division</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>gadc8ccf79f5be15b1ed782efd21b1bd5f</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, std::vector&lt; double &gt; vector_b, int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>Vector_Sum_Reduction</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>gad654c20d23fa2c428619536f6f850651</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>Vector_Min_Reduction</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>ga363773f976560ac01dbbc23baa98ccc0</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>Vector_Max_Reduction</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>ga31e509922e8e5cc38cbd36a38f6e96fe</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, int platform_index=0, int device_index=0)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>Vector_Dot_Product</name>
      <anchorfile>group___vector.html</anchorfile>
      <anchor>gad27fb35fc715e166f6a11336cf293b8a</anchor>
      <arglist>(std::vector&lt; double &gt; vector_a, std::vector&lt; double &gt; vector_b, int platform_index=0, int device_index=0)</arglist>
    </member>
  </compound>
  <compound kind="page">
    <name>index</name>
    <title></title>
    <filename>index.html</filename>
    <docanchor file="index.html">md__mnt_c_Users_Osman_Desktop_Projects_Public_Repos_PySYCL_doc_doxygen_mainpage</docanchor>
  </compound>
</tagfile>
