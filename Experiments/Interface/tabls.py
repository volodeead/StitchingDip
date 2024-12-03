from graphviz import Digraph

# Diagram 1: UML Class Diagram - Main Components
class_diagram = Digraph('class_diagram', format='png')
class_diagram.attr(rankdir='TB')
class_diagram.attr('node', shape='rectangle')

# Nodes representing main classes and their functions
class_diagram.node('PanoramaApp', '''PanoramaApp
- directory_path
- stitching_thread
- stop_event
+ create_main_window()
+ select_directory()
+ start_processing()
+ create_processing_window()
+ stop_processing()
+ run_stitching()
+ create_results_window()
+ create_quality_analysis()
+ run_analysis()
+ create_analysis_window()''')

class_diagram.node('Processing', '''Processing
+ check_gps_and_copy(source_file, destination_dir)
+ remove_metadata(file_path)
+ process_images(source_dir, destination_dir)
+ stitch_image_in_sub_directory(input_directory, output_directory, progress_callback=None)''')

class_diagram.node('Analysis', '''Analysis
+ load_images_in_pairs(directory, progress_callback=None)
+ calculate_metrics()
+ save_metrics_to_json()''')

# Relationships
class_diagram.edge('PanoramaApp', 'Processing', label='calls')
class_diagram.edge('PanoramaApp', 'Analysis', label='calls')

# Render UML Class Diagram
class_diagram.render('/mnt/data/uml_class_diagram')

# Diagram 2: UML Sequence Diagram - Process Flow
sequence_diagram = Digraph('sequence_diagram', format='png')
sequence_diagram.attr(rankdir='LR')
sequence_diagram.attr('node', shape='rectangle')

# Nodes representing components in the process
sequence_diagram.node('User', 'User')
sequence_diagram.node('PanoramaApp', 'PanoramaApp')
sequence_diagram.node('Processing', 'Processing')
sequence_diagram.node('Analysis', 'Analysis')

# Sequence of operations
sequence_diagram.edge('User', 'PanoramaApp', 'starts PanoramaApp')
sequence_diagram.edge('PanoramaApp', 'Processing', 'select_directory()')
sequence_diagram.edge('PanoramaApp', 'Processing', 'start_processing()')
sequence_diagram.edge('Processing', 'Processing', 'process_images()')
sequence_diagram.edge('Processing', 'Processing', 'stitch_image_in_sub_directory()')
sequence_diagram.edge('PanoramaApp', 'Analysis', 'run_analysis()')
sequence_diagram.edge('Analysis', 'Analysis', 'load_images_in_pairs()')

# Render UML Sequence Diagram
sequence_diagram.render('/mnt/data/uml_sequence_diagram')

# Diagram 3: Activity Diagram - Image Processing Flow
activity_diagram = Digraph('activity_diagram', format='png')
activity_diagram.attr(rankdir='TB')
activity_diagram.attr('node', shape='ellipse')

# Activity nodes
activity_diagram.node('start', 'Start', shape='circle')
activity_diagram.node('get_files', 'Get List of Files')
activity_diagram.node('check_gps', 'Check GPS & Copy')
activity_diagram.node('remove_metadata', 'Remove Metadata')
activity_diagram.node('sort_files', 'Sort Files by Timestamp')
activity_diagram.node('rename_files', 'Rename Files')
activity_diagram.node('stitch_images', 'Stitch Images')
activity_diagram.node('end', 'End', shape='circle')

# Flow between activities
activity_diagram.edge('start', 'get_files')
activity_diagram.edge('get_files', 'check_gps')
activity_diagram.edge('check_gps', 'remove_metadata')
activity_diagram.edge('remove_metadata', 'sort_files')
activity_diagram.edge('sort_files', 'rename_files')
activity_diagram.edge('rename_files', 'stitch_images')
activity_diagram.edge('stitch_images', 'end')

# Render Activity Diagram
activity_diagram.render('/mnt/data/activity_diagram')

# Diagram 4: Component Interaction Diagram - Threading and Processing
interaction_diagram = Digraph('interaction_diagram', format='png')
interaction_diagram.attr(rankdir='LR')
interaction_diagram.attr('node', shape='box')

# Nodes representing components involved in threading and processing
interaction_diagram.node('GUI', 'PanoramaApp GUI')
interaction_diagram.node('ThreadPool', 'ThreadPoolExecutor')
interaction_diagram.node('Stitching', 'Stitching Functions')
interaction_diagram.node('Analysis', 'Analysis Functions')

# Interaction flows
interaction_diagram.edge('GUI', 'ThreadPool', 'start stitching in new thread')
interaction_diagram.edge('ThreadPool', 'Stitching', 'execute stitching tasks')
interaction_diagram.edge('ThreadPool', 'Analysis', 'run quality analysis in parallel')

# Render Component Interaction Diagram
interaction_diagram.render('/mnt/data/component_interaction_diagram')

# Returning paths to generated diagrams
{
    "uml_class_diagram": "/mnt/data/uml_class_diagram.png",
    "uml_sequence_diagram": "/mnt/data/uml_sequence_diagram.png",
    "activity_diagram": "/mnt/data/activity_diagram.png",
    "component_interaction_diagram": "/mnt/data/component_interaction_diagram.png"
}
